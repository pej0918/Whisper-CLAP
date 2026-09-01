import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from train_whisper_projector_v2 import (
    MathSpeechDataset,
    DataCollatorSpeechSeq2SeqWithClap,
    build_adapter,
    build_pooler,
    compute_alignment_loss,
    set_seed,
    make_split,
    validate,
)


def load_clap_embeddings(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        if "embeddings" in obj:
            obj = obj["embeddings"]
        elif "text_emb" in obj:
            obj = obj["text_emb"]
        elif "clap_emb" in obj:
            obj = obj["clap_emb"]
        else:
            raise ValueError(f"Unknown CLAP embedding dict keys: {obj.keys()}")

    return obj.float()


class WhisperSemanticASRPosition(nn.Module):
    def __init__(
        self,
        whisper_name="openai/whisper-base",
        clap_dim=512,
        adapter_type="residual_mlp",
        pool_type="mean",
        adapter_position="post_encoder",
        encoder_layer=6,
        adapter_bottleneck=256,
        dropout=0.1,
        adapter_scale_init=0.01,
        freeze_whisper=True,
    ):
        super().__init__()

        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        self.whisper.config.use_cache = False

        hidden_dim = self.whisper.config.d_model

        self.adapter_position = adapter_position
        self.encoder_layer = encoder_layer

        self.adapter = build_adapter(
            adapter_type=adapter_type,
            hidden_dim=hidden_dim,
            bottleneck_dim=adapter_bottleneck,
            dropout=dropout,
            scale_init=adapter_scale_init,
        )

        self.pooler = build_pooler(pool_type, hidden_dim)

        self.align_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, clap_dim),
        )

        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad = False

        for p in self.adapter.parameters():
            p.requires_grad = True
        for p in self.pooler.parameters():
            p.requires_grad = True
        for p in self.align_head.parameters():
            p.requires_grad = True

    def _run_encoder_with_layer_adapter(self, input_features):
        layers = self.whisper.model.encoder.layers
        layer_idx = min(max(self.encoder_layer, 0), len(layers) - 1)

        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                h = self.adapter(h)
                return (h,) + output[1:]
            else:
                return self.adapter(output)

        handle = layers[layer_idx].register_forward_hook(hook_fn)

        try:
            encoder_outputs = self.whisper.model.encoder(input_features)
        finally:
            handle.remove()

        return encoder_outputs.last_hidden_state

    def encode_with_adapter(self, input_features, return_original=False):
        if self.adapter_position == "post_encoder":
            encoder_outputs = self.whisper.model.encoder(input_features)
            h_original = encoder_outputs.last_hidden_state
            h_adapted = self.adapter(h_original)

        elif self.adapter_position == "encoder_layer":
            with torch.no_grad():
                h_original = self.whisper.model.encoder(input_features).last_hidden_state.detach()

            h_adapted = self._run_encoder_with_layer_adapter(input_features)

        elif self.adapter_position == "both":
            with torch.no_grad():
                h_original = self.whisper.model.encoder(input_features).last_hidden_state.detach()

            h_mid = self._run_encoder_with_layer_adapter(input_features)
            h_adapted = self.adapter(h_mid)

        else:
            raise ValueError(f"Unknown adapter_position: {self.adapter_position}")

        if return_original:
            return h_adapted, h_original

        return h_adapted

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
        h_adapted, h_original = self.encode_with_adapter(
            input_features,
            return_original=True,
        )

        adapted_encoder_outputs = BaseModelOutput(last_hidden_state=h_adapted)

        outputs = self.whisper(
            encoder_outputs=adapted_encoder_outputs,
            labels=labels,
            return_dict=True,
        )

        ce_loss = outputs.loss
        total_loss = ce_loss

        hidden_loss = F.mse_loss(h_adapted, h_original.detach())

        if lambda_hidden > 0:
            total_loss = total_loss + lambda_hidden * hidden_loss

        align_loss = None
        loss_parts = {
            "cosine": torch.tensor(0.0, device=ce_loss.device),
            "mse": torch.tensor(0.0, device=ce_loss.device),
            "clip": torch.tensor(0.0, device=ce_loss.device),
        }

        if clap_emb is not None and lambda_align > 0 and align_loss_type != "none":
            pooled = self.pooler(h_adapted)
            z = self.align_head(pooled)

            align_loss, loss_parts = compute_alignment_loss(
                z=z,
                target=clap_emb,
                loss_type=align_loss_type,
                temperature=temperature,
                lambda_cosine=lambda_cosine,
                lambda_mse=lambda_mse,
                lambda_clip=lambda_clip,
            )

            total_loss = total_loss + lambda_align * align_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "hidden_loss": hidden_loss.detach(),
            "align_loss": align_loss.detach() if align_loss is not None else None,
            "cosine_loss": loss_parts["cosine"].detach(),
            "mse_loss": loss_parts["mse"].detach(),
            "clip_loss": loss_parts["clip"].detach(),
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(self, input_features, **kwargs):
        h_adapted = self.encode_with_adapter(input_features)
        adapted_encoder_outputs = BaseModelOutput(last_hidden_state=h_adapted)

        pred_ids = self.whisper.generate(
            encoder_outputs=adapted_encoder_outputs,
            **kwargs,
        )

        return pred_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--clap_emb_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt")
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")

    parser.add_argument(
        "--adapter_type",
        type=str,
        default="residual_mlp",
        choices=["none", "linear_residual", "residual_mlp", "bottleneck", "gated", "conv1d"],
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        default="mean",
        choices=["mean", "cls", "attn"],
    )

    parser.add_argument(
        "--adapter_position",
        type=str,
        default="post_encoder",
        choices=["post_encoder", "encoder_layer", "both"],
    )
    parser.add_argument("--encoder_layer", type=int, default=6)

    parser.add_argument("--adapter_bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter_scale_init", type=float, default=0.01)

    parser.add_argument(
        "--align_loss_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "mse", "clip", "cosine_clip", "cosine_mse", "all"],
    )
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--lambda_hidden", type=float, default=0.1)
    parser.add_argument("--lambda_cosine", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_whisper", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("adapter_type:", args.adapter_type)
    print("pool_type:", args.pool_type)
    print("adapter_position:", args.adapter_position)
    print("encoder_layer:", args.encoder_layer)
    print("align_loss_type:", args.align_loss_type)
    print("lambda_align:", args.lambda_align)
    print("lambda_hidden:", args.lambda_hidden)

    df = pd.read_excel(args.excel_path)
    n = len(df)

    train_idx, valid_idx, test_idx = make_split(n, seed=args.seed)

    split_path = os.path.join(args.save_dir, "split_indices.pt")
    torch.save(
        {
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "seed": args.seed,
        },
        split_path,
    )

    processor = WhisperProcessor.from_pretrained(
        args.whisper_name,
        language="en",
        task="transcribe",
    )

    clap_embs = None
    clap_dim = 512

    use_align = args.lambda_align > 0 and args.align_loss_type != "none"

    if use_align:
        clap_embs = load_clap_embeddings(args.clap_emb_path)
        print("loaded clap embs:", clap_embs.shape)

        if clap_embs.shape[0] != n:
            raise ValueError(f"CLAP embedding count mismatch: {clap_embs.shape[0]} vs {n}")

        clap_dim = clap_embs.shape[-1]
    else:
        print("Alignment disabled. CE-only training.")

    train_dataset = MathSpeechDataset(
        df=df,
        indices=train_idx,
        audio_dir=args.audio_dir,
        processor=processor,
        clap_embs=clap_embs,
    )

    valid_dataset = MathSpeechDataset(
        df=df,
        indices=valid_idx,
        audio_dir=args.audio_dir,
        processor=processor,
        clap_embs=clap_embs,
    )

    collator = DataCollatorSpeechSeq2SeqWithClap(processor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = WhisperSemanticASRPosition(
        whisper_name=args.whisper_name,
        clap_dim=clap_dim,
        adapter_type=args.adapter_type,
        pool_type=args.pool_type,
        adapter_position=args.adapter_position,
        encoder_layer=args.encoder_layer,
        adapter_bottleneck=args.adapter_bottleneck,
        dropout=args.dropout,
        adapter_scale_init=args.adapter_scale_init,
        freeze_whisper=args.freeze_whisper,
    ).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe",
    )
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("total params:", sum(p.numel() for p in model.parameters()))
    print("trainable params:", sum(p.numel() for p in trainable_params))
    print("clap_dim:", clap_dim)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_valid_loss = float("inf")
    log_rows = []
    log_path = os.path.join(args.save_dir, "train_log.csv")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_hidden = 0.0
        total_align = 0.0
        total_cos = 0.0
        total_mse = 0.0
        total_clip = 0.0
        align_count = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")

        for batch in pbar:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            clap_emb = batch.get("clap_emb", None)
            if clap_emb is not None:
                clap_emb = clap_emb.to(device)

            out = model(
                input_features=input_features,
                labels=labels,
                clap_emb=clap_emb,
                lambda_align=args.lambda_align,
                lambda_hidden=args.lambda_hidden,
                align_loss_type=args.align_loss_type,
                temperature=args.temperature,
                lambda_cosine=args.lambda_cosine,
                lambda_mse=args.lambda_mse,
                lambda_clip=args.lambda_clip,
            )

            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += out["loss"].item()
            total_ce += out["ce_loss"].item()
            total_hidden += out["hidden_loss"].item()
            total_cos += out["cosine_loss"].item()
            total_mse += out["mse_loss"].item()
            total_clip += out["clip_loss"].item()

            if out["align_loss"] is not None:
                total_align += out["align_loss"].item()
                align_count += 1

            pbar.set_postfix(
                loss=f"{out['loss'].item():.4f}",
                ce=f"{out['ce_loss'].item():.4f}",
                hidden=f"{out['hidden_loss'].item():.4f}",
                align=f"{out['align_loss'].item():.4f}" if out["align_loss"] is not None else "0.0000",
                cos=f"{out['cosine_loss'].item():.4f}",
                clip=f"{out['clip_loss'].item():.4f}",
            )

        denom = max(len(train_loader), 1)

        train_stats = {
            "loss": total_loss / denom,
            "ce": total_ce / denom,
            "hidden": total_hidden / denom,
            "align": total_align / max(align_count, 1),
            "cosine": total_cos / denom,
            "mse": total_mse / denom,
            "clip": total_clip / denom,
        }

        valid_stats = validate(model, valid_loader, device, args)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_stats['loss']:.4f}, "
            f"train_ce={train_stats['ce']:.4f}, "
            f"train_hidden={train_stats['hidden']:.4f}, "
            f"train_align={train_stats['align']:.4f} | "
            f"valid_loss={valid_stats['loss']:.4f}, "
            f"valid_ce={valid_stats['ce']:.4f}, "
            f"valid_hidden={valid_stats['hidden']:.4f}, "
            f"valid_align={valid_stats['align']:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_hidden": train_stats["hidden"],
            "train_align": train_stats["align"],
            "train_cosine": train_stats["cosine"],
            "train_mse": train_stats["mse"],
            "train_clip": train_stats["clip"],
            "valid_loss": valid_stats["loss"],
            "valid_ce": valid_stats["ce"],
            "valid_hidden": valid_stats["hidden"],
            "valid_align": valid_stats["align"],
            "valid_cosine": valid_stats["cosine"],
            "valid_mse": valid_stats["mse"],
            "valid_clip": valid_stats["clip"],
        })

        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "clap_dim": clap_dim,
            "epoch": epoch,
            "train_stats": train_stats,
            "valid_stats": valid_stats,
        }

        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if valid_stats["loss"] < best_valid_loss:
            best_valid_loss = valid_stats["loss"]
            ckpt["best_valid_loss"] = best_valid_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print("saved best:", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()