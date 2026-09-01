import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

TARGET_SR = 16000


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_audio(path):
    """Load a complete MathSpeech utterance without depending on openai-whisper."""
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        wav = f.read(dtype="float32", always_2d=True)

    if wav.shape[0] == 0:
        raise RuntimeError(f"Empty audio: {path}")

    wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        wav = torch.from_numpy(wav)
        wav = AF.resample(wav, sr, TARGET_SR)
        wav = wav.numpy()

    wav = wav.astype(np.float32)
    duration = len(wav) / TARGET_SR

    if duration > 30.0:
        raise RuntimeError(f"Audio exceeds 30 sec ({duration:.3f}s): {path}")

    return wav


class MathSpeechDataset(Dataset):
    """
    Reads our source-disjoint split CSV directly.

    Required columns:
      sample_id, audio_path, reference_text

    CLAP embeddings are assumed to follow the original MathSpeech ordering:
      sample_id=1 -> clap_embs[0], ..., sample_id=1101 -> clap_embs[1100].
    """
    def __init__(self, csv_path, processor, clap_embs=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.processor = processor
        self.clap_embs = clap_embs

        required = ["sample_id", "audio_path", "reference_text"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        ids = self.df["sample_id"].astype(int)
        if ids.duplicated().any():
            raise ValueError(f"Duplicate sample_id in {csv_path}")
        if (ids < 1).any():
            raise ValueError(f"sample_id must be 1-based in {csv_path}")

        if clap_embs is not None and int(ids.max()) > len(clap_embs):
            raise ValueError(
                f"sample_id {int(ids.max())} exceeds CLAP embedding count {len(clap_embs)}"
            )

        print(f"{csv_path}: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        sample_id = int(r["sample_id"])
        audio_path = str(r["audio_path"])
        text = str(r["reference_text"])

        wav = load_audio(audio_path)

        input_features = self.processor.feature_extractor(
            wav,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
        ).input_ids[0]

        item = {
            "input_features": input_features,
            "labels": labels,
            "text": text,
            "sample_id": sample_id,
        }

        if self.clap_embs is not None:
            item["clap_emb"] = self.clap_embs[sample_id - 1]

        return item


class DataCollatorSpeechSeq2SeqWithClap:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, batch):
        input_features = torch.stack([b["input_features"] for b in batch], dim=0)

        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": b["labels"]} for b in batch],
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # Same HF Whisper convention used in our M3AV pipeline.
        if (
            labels.shape[1] > 0
            and (labels[:, 0] == self.decoder_start_token_id).all().item()
        ):
            labels = labels[:, 1:]

        out = {
            "input_features": input_features,
            "labels": labels,
            "texts": [b["text"] for b in batch],
            "sample_ids": [b["sample_id"] for b in batch],
        }

        if "clap_emb" in batch[0]:
            out["clap_emb"] = torch.stack([b["clap_emb"] for b in batch], dim=0)

        return out


class IdentityAdapter(nn.Module):
    def forward(self, h):
        return h


class LinearResidualAdapter(nn.Module):
    def __init__(self, hidden_dim, scale_init=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, h):
        return h + self.scale * self.proj(self.norm(h))


class ResidualMLPAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=256, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, h):
        return h + self.scale * self.net(h)


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=128, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, h):
        x = self.norm(h)
        x = self.down(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up(x)
        return h + self.scale * x


class GatedAdapter(nn.Module):
    """Team Ours gated residual adapter."""
    def __init__(self, hidden_dim, bottleneck_dim=256, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, h):
        x = self.norm(h)
        delta = self.delta(x)
        gate = self.gate(x)
        return h + self.scale * gate * delta


class Conv1DAdapter(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, h):
        x = self.norm(h).transpose(1, 2)
        x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x).transpose(1, 2)
        return h + self.scale * x


def build_adapter(adapter_type, hidden_dim, bottleneck_dim, dropout, scale_init):
    if adapter_type == "none":
        return IdentityAdapter()
    if adapter_type == "linear_residual":
        return LinearResidualAdapter(hidden_dim, scale_init)
    if adapter_type == "residual_mlp":
        return ResidualMLPAdapter(hidden_dim, bottleneck_dim, dropout, scale_init)
    if adapter_type == "bottleneck":
        return BottleneckAdapter(hidden_dim, bottleneck_dim, dropout, scale_init)
    if adapter_type == "gated":
        return GatedAdapter(hidden_dim, bottleneck_dim, dropout, scale_init)
    if adapter_type == "conv1d":
        return Conv1DAdapter(hidden_dim, 3, dropout, scale_init)
    raise ValueError(adapter_type)


class MeanPooler(nn.Module):
    def forward(self, h):
        return h.mean(dim=1)


class CLSPooler(nn.Module):
    def forward(self, h):
        # This preserves the teammate implementation: first encoder time step.
        return h[:, 0]


class AttentionPooler(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, h):
        w = torch.softmax(self.score(h).squeeze(-1), dim=-1)
        return torch.sum(h * w.unsqueeze(-1), dim=1)


def build_pooler(pool_type, hidden_dim):
    if pool_type == "mean":
        return MeanPooler()
    if pool_type == "cls":
        return CLSPooler()
    if pool_type == "attn":
        return AttentionPooler(hidden_dim)
    raise ValueError(pool_type)


def cosine_alignment_loss(z, target):
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1.0 - (z * target).sum(dim=-1)).mean()


def mse_alignment_loss(z, target):
    return F.mse_loss(z, target)


def clip_contrastive_loss(z, target, temperature=0.07):
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (z @ target.t()) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_alignment_loss(
    z,
    target,
    loss_type="cosine",
    temperature=0.07,
    lambda_cosine=1.0,
    lambda_mse=1.0,
    lambda_clip=1.0,
):
    zero = z.new_tensor(0.0)
    if loss_type == "none":
        return zero, {"cosine": zero.detach(), "mse": zero.detach(), "clip": zero.detach()}

    cos = cosine_alignment_loss(z, target)
    mse = mse_alignment_loss(z, target)
    clip = clip_contrastive_loss(z, target, temperature)

    if loss_type == "cosine":
        total = lambda_cosine * cos
    elif loss_type == "mse":
        total = lambda_mse * mse
    elif loss_type == "clip":
        total = lambda_clip * clip
    elif loss_type == "cosine_clip":
        total = lambda_cosine * cos + lambda_clip * clip
    elif loss_type == "cosine_mse":
        total = lambda_cosine * cos + lambda_mse * mse
    elif loss_type == "all":
        total = lambda_cosine * cos + lambda_mse * mse + lambda_clip * clip
    else:
        raise ValueError(loss_type)

    return total, {"cosine": cos.detach(), "mse": mse.detach(), "clip": clip.detach()}


class WhisperSemanticASR(nn.Module):
    """Team Ours model definition, using HuggingFace Whisper."""
    def __init__(
        self,
        whisper_name="openai/whisper-base",
        clap_dim=512,
        adapter_type="gated",
        pool_type="cls",
        adapter_bottleneck=256,
        dropout=0.1,
        adapter_scale_init=0.01,
        freeze_whisper=True,
    ):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        self.whisper.config.use_cache = False
        hidden_dim = self.whisper.config.d_model

        self.adapter = build_adapter(
            adapter_type, hidden_dim, adapter_bottleneck, dropout, adapter_scale_init
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

    def encode_with_adapter(self, input_features, return_original=False):
        encoder_outputs = self.whisper.model.encoder(input_features)
        h_original = encoder_outputs.last_hidden_state
        h_adapted = self.adapter(h_original)
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
        parts = {"cosine": zero.detach(), "mse": zero.detach(), "clip": zero.detach()}

        if clap_emb is not None and lambda_align > 0 and align_loss_type != "none":
            pooled = self.pooler(h_adapted)
            z = self.align_head(pooled)
            align_loss, parts = compute_alignment_loss(
                z,
                clap_emb,
                align_loss_type,
                temperature,
                lambda_cosine,
                lambda_mse,
                lambda_clip,
            )
            total_loss = total_loss + lambda_align * align_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "hidden_loss": hidden_loss.detach(),
            "align_loss": align_loss.detach() if align_loss is not None else None,
            "cosine_loss": parts["cosine"].detach(),
            "mse_loss": parts["mse"].detach(),
            "clip_loss": parts["clip"].detach(),
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(self, input_features, **kwargs):
        h_adapted = self.encode_with_adapter(input_features)
        return self.whisper.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=h_adapted),
            **kwargs,
        )


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    sums = {"loss": 0.0, "ce": 0.0, "hidden": 0.0, "align": 0.0}
    align_count = 0

    for batch in tqdm(loader, desc="valid"):
        feats = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        clap_emb = batch.get("clap_emb")
        if clap_emb is not None:
            clap_emb = clap_emb.to(device)

        out = model(
            input_features=feats,
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
        sums["loss"] += out["loss"].item()
        sums["ce"] += out["ce_loss"].item()
        sums["hidden"] += out["hidden_loss"].item()
        if out["align_loss"] is not None:
            sums["align"] += out["align_loss"].item()
            align_count += 1

    n = max(len(loader), 1)
    return {
        "loss": sums["loss"] / n,
        "ce": sums["ce"] / n,
        "hidden": sums["hidden"] / n,
        "align": sums["align"] / max(align_count, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--valid_csv", required=True)
    ap.add_argument("--test_csv", required=True)  # saved for reproducibility; not used for training
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--clap_emb_path", default="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt")
    ap.add_argument("--whisper_name", default="openai/whisper-base")

    ap.add_argument("--adapter_type", default="gated", choices=["none", "linear_residual", "residual_mlp", "bottleneck", "gated", "conv1d"])
    ap.add_argument("--pool_type", default="cls", choices=["mean", "cls", "attn"])
    ap.add_argument("--adapter_bottleneck", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--adapter_scale_init", type=float, default=0.01)

    ap.add_argument("--align_loss_type", default="cosine", choices=["none", "cosine", "mse", "clip", "cosine_clip", "cosine_mse", "all"])
    ap.add_argument("--lambda_align", type=float, default=0.05)
    ap.add_argument("--lambda_hidden", type=float, default=0.1)
    ap.add_argument("--lambda_cosine", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_whisper", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Explicitly verify split integrity.
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

    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("split sizes:", {k: len(v) for k, v in ids.items()})
    print("adapter_type:", args.adapter_type)
    print("pool_type:", args.pool_type)
    print("align_loss_type:", args.align_loss_type)
    print("lambda_align:", args.lambda_align)
    print("lambda_hidden:", args.lambda_hidden)

    processor = WhisperProcessor.from_pretrained(
        args.whisper_name, language="English", task="transcribe"
    )

    use_align = args.lambda_align > 0 and args.align_loss_type != "none"
    clap_embs = None
    clap_dim = 512
    if use_align:
        clap_embs = torch_load_compat(args.clap_emb_path, map_location="cpu").float()
        if clap_embs.ndim != 2:
            raise ValueError(f"Expected 2D CLAP embeddings, got {clap_embs.shape}")
        clap_dim = int(clap_embs.shape[-1])
        print("loaded CLAP embeddings:", tuple(clap_embs.shape))
    else:
        print("CLAP alignment disabled")

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
    )

    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_ids
    model.whisper.generation_config.forced_decoder_ids = forced_ids

    collator = DataCollatorSpeechSeq2SeqWithClap(
        processor, model.whisper.config.decoder_start_token_id
    )

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
    print("total params:", sum(p.numel() for p in model.parameters()))
    print("trainable params:", sum(p.numel() for p in trainable))
    print("freeze_whisper:", args.freeze_whisper)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_valid_loss = float("inf")
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "hidden": 0.0, "align": 0.0}
        align_count = 0

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

            pbar.set_postfix(
                loss=f"{out['loss'].item():.4f}",
                ce=f"{out['ce_loss'].item():.4f}",
                hidden=f"{out['hidden_loss'].item():.4f}",
                align=(f"{out['align_loss'].item():.4f}" if out["align_loss"] is not None else "0.0000"),
            )

        n = max(len(train_loader), 1)
        train_stats = {
            "loss": sums["loss"] / n,
            "ce": sums["ce"] / n,
            "hidden": sums["hidden"] / n,
            "align": sums["align"] / max(align_count, 1),
        }
        valid_stats = validate(model, valid_loader, device, args)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_stats['loss']:.4f} ce={train_stats['ce']:.4f} "
            f"hidden={train_stats['hidden']:.4f} align={train_stats['align']:.4f} | "
            f"valid_loss={valid_stats['loss']:.4f} ce={valid_stats['ce']:.4f} "
            f"hidden={valid_stats['hidden']:.4f} align={valid_stats['align']:.4f}"
        )

        rows.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"valid_{k}": v for k, v in valid_stats.items()},
        })
        pd.DataFrame(rows).to_csv(outdir / "train_log.csv", index=False)

        ckpt = {
            "args": vars(args),
            "clap_dim": clap_dim,
            "epoch": epoch,
            "train_stats": train_stats,
            "valid_stats": valid_stats,
            "model_state_dict": model.state_dict(),
            # Also save compact component states so the M3AV-style evaluator pattern can be reused.
            "adapter_state_dict": model.adapter.state_dict(),
            "pooler_state_dict": model.pooler.state_dict(),
            "align_head_state_dict": model.align_head.state_dict(),
        }
        torch.save(ckpt, outdir / "last.pt")

        if valid_stats["loss"] < best_valid_loss:
            best_valid_loss = valid_stats["loss"]
            ckpt["best_valid_loss"] = best_valid_loss
            torch.save(ckpt, outdir / "best.pt")
            print("saved best:", outdir / "best.pt")

    with open(outdir / "training_summary.json", "w") as f:
        json.dump(
            {
                "best_valid_loss": best_valid_loss,
                "epochs": args.epochs,
                "train_samples": len(train_ds),
                "valid_samples": len(valid_ds),
                "test_samples": len(split_dfs["test"]),
                "args": vars(args),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
