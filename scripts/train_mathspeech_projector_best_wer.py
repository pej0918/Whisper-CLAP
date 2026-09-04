import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from jiwer import wer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

from train_mathspeech_projector_source_disjoint_hf import (
    DataCollatorSpeechSeq2SeqWithClap,
    MathSpeechDataset,
    WhisperSemanticASR as BaseWhisperSemanticASR,
    compute_alignment_loss,
    set_seed,
    torch_load_compat,
    validate,
)


class WhisperSemanticASR(BaseWhisperSemanticASR):
    """
    MathSpeech projector model with selectable semantic-alignment objective.

    alignment_mode="absolute": preserves the original implementation exactly.
      L_align = the loss selected by --align_loss_type between the adapted
      representation and the CLAP text embedding.

    alignment_mode="relative": parameter-free relative semantic alignment.
      For cosine alignment, encourage the adapted representation to have
      higher CLAP cosine similarity than the original Whisper representation:

          L_rel = softplus(sim_original - sim_adapted)

      No margin or additional tunable hyperparameter is introduced. The
      original branch is detached so it acts only as a reference.

    alignment_mode="absolute_relational": complementary point-wise and
      structural semantic supervision. The absolute term anchors each adapted
      utterance to its Lecture-CLAP text target, while the relational term
      matches pairwise semantic similarities among samples directly in the
      decoder-facing pooled Whisper representation:

          L_align = 0.5 * L_absolute + 0.5 * L_relational

      where L_relational is the off-diagonal MSE between the student and
      teacher cosine-similarity matrices. This mode introduces no additional
      tunable loss weight and currently requires --align_loss_type cosine.
    """

    def __init__(self, *args, alignment_mode="absolute", **kwargs):
        super().__init__(*args, **kwargs)
        if alignment_mode not in {"absolute", "relative", "absolute_relational"}:
            raise ValueError(f"Unknown alignment_mode: {alignment_mode}")
        self.alignment_mode = alignment_mode

    @staticmethod
    def relational_semantic_loss(student_repr, teacher_emb):
        """Match pairwise semantic geometry without projecting spaces together.

        student_repr: [B, d] pooled adapted Whisper representations.
        teacher_emb:  [B, d_t] frozen Lecture-CLAP text embeddings.

        The diagonal is excluded because self-similarity is always 1 and gives
        no useful relational supervision. For a singleton batch there are no
        valid pairs, so return a differentiable zero.
        """
        if student_repr.size(0) < 2:
            return student_repr.sum() * 0.0

        student_norm = F.normalize(student_repr, dim=-1)
        teacher_norm = F.normalize(teacher_emb.detach(), dim=-1)

        student_sim = student_norm @ student_norm.transpose(0, 1)
        teacher_sim = teacher_norm @ teacher_norm.transpose(0, 1)

        bsz = student_sim.size(0)
        off_diag = ~torch.eye(bsz, dtype=torch.bool, device=student_sim.device)
        return F.mse_loss(student_sim[off_diag], teacher_sim[off_diag])

    def forward(
        self,
        input_features,
        labels,
        clap_emb=None,
        lambda_align=0.1,
        lambda_hidden=0.1,
        align_loss_type="cosine",
        temperature=0.07,
        lambda_cosine=1.0,
        lambda_mse=1.0,
        lambda_clip=1.0,
    ):
        h_adapted, h_original = self.encode_with_adapter(input_features, True)
        outputs = self.whisper(
            encoder_outputs=BaseModelOutput(last_hidden_state=h_adapted),
            labels=labels,
            return_dict=True,
        )

        ce_loss = outputs.loss
        hidden_loss = F.mse_loss(h_adapted, h_original.detach())
        total_loss = ce_loss + lambda_hidden * hidden_loss

        zero = ce_loss.new_tensor(0.0)
        align_loss = None
        absolute_align_loss = None
        relational_align_loss = None
        parts = {
            "cosine": zero.detach(),
            "mse": zero.detach(),
            "clip": zero.detach(),
        }

        if clap_emb is not None and lambda_align > 0 and align_loss_type != "none":
            pooled_adapted = self.pooler(h_adapted)
            z_adapted = self.align_head(pooled_adapted)

            if self.alignment_mode == "absolute":
                # Original behavior: keep all existing absolute alignment losses.
                align_loss, parts = compute_alignment_loss(
                    z_adapted,
                    clap_emb,
                    align_loss_type,
                    temperature,
                    lambda_cosine,
                    lambda_mse,
                    lambda_clip,
                )
                absolute_align_loss = align_loss

            elif self.alignment_mode == "relative":
                # Relative semantic alignment is intentionally cosine-only.
                # This keeps the objective margin-free and introduces no new
                # hyperparameter beyond the existing lambda_align/lambda_cosine.
                if align_loss_type != "cosine":
                    raise ValueError(
                        "alignment_mode='relative' currently requires "
                        "--align_loss_type cosine"
                    )

                pooled_original = self.pooler(h_original.detach())
                z_original = self.align_head(pooled_original).detach()

                z_adapted_norm = F.normalize(z_adapted, dim=-1)
                z_original_norm = F.normalize(z_original, dim=-1)
                target_norm = F.normalize(clap_emb, dim=-1)

                sim_adapted = (z_adapted_norm * target_norm).sum(dim=-1)
                sim_original = (z_original_norm * target_norm).sum(dim=-1)

                # Smooth, margin-free ranking objective:
                #   lower loss when adapted semantics are better than original.
                align_loss = lambda_cosine * F.softplus(
                    sim_original - sim_adapted
                ).mean()

                # Keep cosine_loss as an interpretable diagnostic compatible
                # with the existing logs: absolute cosine distance of adapted z.
                parts["cosine"] = (1.0 - sim_adapted).mean().detach()

            else:
                # Absolute + relational semantic alignment.
                # Point-wise absolute supervision still uses AlignHead so the
                # trainable parameter count and original semantic anchoring are
                # preserved. In parallel, relational supervision bypasses the
                # AlignHead and acts directly on the pooled representation that
                # is derived from the decoder-facing h_adapted.
                if align_loss_type != "cosine":
                    raise ValueError(
                        "alignment_mode='absolute_relational' currently requires "
                        "--align_loss_type cosine"
                    )

                absolute_align_loss, parts = compute_alignment_loss(
                    z_adapted,
                    clap_emb,
                    "cosine",
                    temperature,
                    lambda_cosine,
                    lambda_mse,
                    lambda_clip,
                )
                relational_align_loss = lambda_cosine * self.relational_semantic_loss(
                    pooled_adapted,
                    clap_emb,
                )

                # Equal, fixed composition: no extra mixing hyperparameter.
                align_loss = 0.5 * (
                    absolute_align_loss + relational_align_loss
                )

            total_loss = total_loss + lambda_align * align_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "hidden_loss": hidden_loss.detach(),
            "align_loss": align_loss.detach() if align_loss is not None else None,
            "absolute_align_loss": (
                absolute_align_loss.detach() if absolute_align_loss is not None else None
            ),
            "relational_align_loss": (
                relational_align_loss.detach() if relational_align_loss is not None else None
            ),
            "cosine_loss": parts["cosine"].detach(),
            "mse_loss": parts["mse"].detach(),
            "clip_loss": parts["clip"].detach(),
            "logits": outputs.logits,
        }


def normalize_text(tokenizer, text):
    text = str(text).strip()
    if hasattr(tokenizer, "normalize"):
        return tokenizer.normalize(text).strip()
    if hasattr(tokenizer, "_normalize"):
        return tokenizer._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found.")


@torch.no_grad()
def evaluate_validation_wer(model, loader, processor, device, num_beams, max_new_tokens):
    model.eval()
    refs, hyps = [], []

    for batch in tqdm(loader, desc="valid_wer"):
        feats = batch["input_features"].to(device)
        pred_ids = model.generate(
            feats,
            num_beams=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        for ref, hyp in zip(batch["texts"], pred_text):
            ref = normalize_text(processor.tokenizer, ref)
            hyp = normalize_text(processor.tokenizer, hyp)
            if ref:
                refs.append(ref)
                hyps.append(hyp)

    if not refs:
        return float("inf")
    return wer(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--valid_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--clap_emb_path", default="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt")
    ap.add_argument("--whisper_name", default="openai/whisper-base")

    ap.add_argument("--adapter_type", default="gated", choices=["none", "linear_residual", "residual_mlp", "bottleneck", "gated", "conv1d"])
    ap.add_argument("--pool_type", default="cls", choices=["mean", "cls", "attn"])
    ap.add_argument("--adapter_bottleneck", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--adapter_scale_init", type=float, default=0.01)

    ap.add_argument("--align_loss_type", default="cosine", choices=["none", "cosine", "mse", "clip", "cosine_clip", "cosine_mse", "all"])
    ap.add_argument(
        "--alignment_mode",
        default="absolute",
        choices=["absolute", "relative", "absolute_relational"],
        help=(
            "absolute: original alignment implementation; "
            "relative: margin-free softplus(sim_original - sim_adapted) "
            "semantic alignment; "
            "absolute_relational: equal-weight point-wise cosine alignment + "
            "pairwise semantic-geometry matching. The latter two modes require "
            "--align_loss_type cosine"
        ),
    )
    ap.add_argument("--lambda_align", type=float, default=0.05)
    ap.add_argument("--lambda_hidden", type=float, default=0.1)
    ap.add_argument("--lambda_cosine", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_whisper", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--selection_num_beams", type=int, default=5)
    ap.add_argument("--selection_max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    if (
        args.alignment_mode in {"relative", "absolute_relational"}
        and args.align_loss_type not in {"none", "cosine"}
    ):
        ap.error(
            f"--alignment_mode {args.alignment_mode} requires "
            "--align_loss_type cosine (or none)"
        )

    set_seed(args.seed)
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Preserve the source-disjoint integrity checks used by the teammate script.
    split_dfs = {
        "train": pd.read_csv(args.train_csv),
        "valid": pd.read_csv(args.valid_csv),
        "test": pd.read_csv(args.test_csv),
    }
    ids = {k: set(v["sample_id"].astype(int)) for k, v in split_dfs.items()}
    if ids["train"] & ids["valid"] or ids["train"] & ids["test"] or ids["valid"] & ids["test"]:
        raise ValueError("Sample overlap across splits")
    if "source" in split_dfs["train"].columns:
        src = {k: set(v["source"].astype(str)) for k, v in split_dfs.items()}
        overlaps = (
            len(src["train"] & src["valid"]),
            len(src["train"] & src["test"]),
            len(src["valid"] & src["test"]),
        )
        print("source overlap train/valid, train/test, valid/test:", overlaps)
        if any(overlaps):
            raise ValueError(f"Source overlap detected: {overlaps}")

    split_indices = {
        "train_idx": [x - 1 for x in split_dfs["train"]["sample_id"].astype(int).tolist()],
        "valid_idx": [x - 1 for x in split_dfs["valid"]["sample_id"].astype(int).tolist()],
        "test_idx": [x - 1 for x in split_dfs["test"]["sample_id"].astype(int).tolist()],
        "seed": args.seed,
    }
    torch.save(split_indices, outdir / "split_indices.pt")

    processor = WhisperProcessor.from_pretrained(args.whisper_name, language="English", task="transcribe")

    use_align = args.lambda_align > 0 and args.align_loss_type != "none"
    clap_embs = None
    clap_dim = 512
    if use_align:
        clap_embs = torch_load_compat(args.clap_emb_path, map_location="cpu").float()
        if clap_embs.ndim != 2:
            raise ValueError(f"Expected 2D CLAP embeddings, got {clap_embs.shape}")
        clap_dim = int(clap_embs.shape[-1])
        print("loaded CLAP embeddings:", tuple(clap_embs.shape))

    train_ds = MathSpeechDataset(args.train_csv, processor, clap_embs)
    valid_ds = MathSpeechDataset(args.valid_csv, processor, clap_embs)

    model = WhisperSemanticASR(
        whisper_name=args.whisper_name,
        clap_dim=clap_dim,
        adapter_type=args.adapter_type,
        pool_type=args.pool_type,
        adapter_bottleneck=args.adapter_bottleneck,
        dropout=args.dropout,
        adapter_scale_init=args.adapter_scale_init,
        freeze_whisper=args.freeze_whisper,
        alignment_mode=args.alignment_mode,
    )

    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_ids
    model.whisper.generation_config.forced_decoder_ids = forced_ids

    collator = DataCollatorSpeechSeq2SeqWithClap(processor, model.whisper.config.decoder_start_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())

    print("=" * 70)
    print("CLAP PROJECTOR TRAINING — BEST VALIDATION WER SELECTION")
    print("freeze_whisper       :", args.freeze_whisper)
    print("alignment_mode       :", args.alignment_mode)
    print("align_loss_type      :", args.align_loss_type)
    if args.alignment_mode == "absolute_relational":
        print("align composition    : 0.5 * absolute + 0.5 * relational")
    print("selection_num_beams  :", args.selection_num_beams)
    print("selection metric     : valid_wer")
    print("total params         :", total_count)
    print("trainable params     :", trainable_count)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_valid_wer = float("inf")
    best_valid_loss = None
    best_epoch = None
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "loss": 0.0,
            "ce": 0.0,
            "hidden": 0.0,
            "align": 0.0,
            "absolute_align": 0.0,
            "relational_align": 0.0,
        }
        align_count = 0
        absolute_count = 0
        relational_count = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for batch in pbar:
            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            clap_emb = batch.get("clap_emb")
            if clap_emb is not None:
                clap_emb = clap_emb.to(device)

            out = model(
                feats,
                labels,
                clap_emb=clap_emb,
                lambda_align=args.lambda_align,
                lambda_hidden=args.lambda_hidden,
                align_loss_type=args.align_loss_type,
                temperature=args.temperature,
                lambda_cosine=args.lambda_cosine,
                lambda_mse=args.lambda_mse,
                lambda_clip=args.lambda_clip,
            )

            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            sums["loss"] += out["loss"].item()
            sums["ce"] += out["ce_loss"].item()
            sums["hidden"] += out["hidden_loss"].item()
            if out["align_loss"] is not None:
                sums["align"] += out["align_loss"].item()
                align_count += 1
            if out["absolute_align_loss"] is not None:
                sums["absolute_align"] += out["absolute_align_loss"].item()
                absolute_count += 1
            if out["relational_align_loss"] is not None:
                sums["relational_align"] += out["relational_align_loss"].item()
                relational_count += 1

            pbar.set_postfix(
                loss=f"{out['loss'].item():.4f}",
                ce=f"{out['ce_loss'].item():.4f}",
            )

        n = max(len(train_loader), 1)
        train_stats = {
            "loss": sums["loss"] / n,
            "ce": sums["ce"] / n,
            "hidden": sums["hidden"] / n,
            "align": sums["align"] / max(align_count, 1),
            "absolute_align": sums["absolute_align"] / max(absolute_count, 1),
            "relational_align": sums["relational_align"] / max(relational_count, 1),
        }
        valid_stats = validate(model, valid_loader, device, args)
        valid_wer = evaluate_validation_wer(
            model,
            valid_loader,
            processor,
            device,
            args.selection_num_beams,
            args.selection_max_new_tokens,
        )

        print(
            f"[epoch {epoch}] train_loss={train_stats['loss']:.4f} "
            f"valid_loss={valid_stats['loss']:.4f} valid_wer={valid_wer:.6f}"
        )
        if args.alignment_mode == "absolute_relational":
            print(
                f"           train_align={train_stats['align']:.4f} "
                f"absolute={train_stats['absolute_align']:.4f} "
                f"relational={train_stats['relational_align']:.4f}"
            )

        rows.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"valid_{k}": v for k, v in valid_stats.items()},
            "valid_wer": valid_wer,
        })
        pd.DataFrame(rows).to_csv(outdir / "train_log.csv", index=False)

        ckpt = {
            "args": vars(args),
            "clap_dim": clap_dim,
            "epoch": epoch,
            "train_stats": train_stats,
            "valid_stats": valid_stats,
            "valid_wer": valid_wer,
            "selection_metric": "valid_wer",
            "selection_num_beams": args.selection_num_beams,
            "trainable_params": trainable_count,
            "model_state_dict": model.state_dict(),
            "adapter_state_dict": model.adapter.state_dict(),
            "pooler_state_dict": model.pooler.state_dict(),
            "align_head_state_dict": model.align_head.state_dict(),
        }
        torch.save(ckpt, outdir / "last.pt")

        if valid_wer < best_valid_wer:
            best_valid_wer = valid_wer
            best_valid_loss = valid_stats["loss"]
            best_epoch = epoch
            ckpt["best_valid_wer"] = best_valid_wer
            ckpt["best_valid_loss_at_best_wer"] = best_valid_loss
            torch.save(ckpt, outdir / "best.pt")
            print("saved best valid-WER checkpoint:", outdir / "best.pt")

    with open(outdir / "training_summary.json", "w") as f:
        json.dump(
            {
                "best_valid_wer": best_valid_wer,
                "best_dev_wer": best_valid_wer,
                "best_valid_loss_at_best_wer": best_valid_loss,
                "best_epoch": best_epoch,
                "selection_metric": "valid_wer",
                "selection_num_beams": args.selection_num_beams,
                "selection_max_new_tokens": args.selection_max_new_tokens,
                "epochs": args.epochs,
                "train_samples": len(train_ds),
                "valid_samples": len(valid_ds),
                "test_samples": len(split_dfs["test"]),
                "trainable_params": trainable_count,
                "trainable_params_M": trainable_count / 1e6,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    print("best_epoch    :", best_epoch)
    print("best_valid_wer:", best_valid_wer)


if __name__ == "__main__":
    main()
