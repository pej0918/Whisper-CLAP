import os
import json
import time
import random
import argparse
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
from jiwer import wer, cer

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


def load_segment(path, start, end):
    start = float(start)
    end = float(end)
    if end <= start:
        raise ValueError(f"Invalid segment: {path}, {start}, {end}")

    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        start_frame = int(round(start * sr))
        n_frames = int(round((end - start) * sr))
        f.seek(start_frame)
        wav = f.read(frames=n_frames, dtype="float32", always_2d=True)

    if wav.shape[0] == 0:
        raise RuntimeError(f"Empty audio: {path} [{start}, {end}]")

    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = torch.from_numpy(wav)
        wav = AF.resample(wav, sr, TARGET_SR)
        wav = wav.numpy()
    return wav.astype(np.float32)


def load_clap_embedding_store(path):
    obj = torch_load_compat(path, map_location="cpu")
    if not isinstance(obj, dict) or "embeddings" not in obj or "sample_ids" not in obj:
        raise ValueError(
            "M3AV CLAP embedding file must be produced by "
            "precompute_m3av_clap_text_emb.py and contain embeddings + sample_ids."
        )
    embs = obj["embeddings"].float().contiguous()
    sample_ids = [str(x) for x in obj["sample_ids"]]
    if len(sample_ids) != embs.shape[0]:
        raise ValueError("CLAP embedding file has mismatched sample_ids/embeddings lengths.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_ids in CLAP embedding file.")
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    meta = {k: v for k, v in obj.items() if k not in ("embeddings", "sample_ids")}
    meta["embedding_dim"] = int(embs.shape[-1])
    meta["num_embeddings"] = int(embs.shape[0])
    return embs, id_to_idx, meta


class M3AVSemanticDataset(Dataset):
    def __init__(self, manifest, processor, clap_emb_path=None, max_samples=None):
        self.df = pd.read_csv(manifest)
        required = ["sample_id", "audio_path", "start", "end", "duration", "text_spoken"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing manifest columns: {missing}")

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].copy()
        self.df = self.df.reset_index(drop=True)

        n_long = int((self.df["duration"].astype(float) > 30.0).sum())
        if n_long:
            raise RuntimeError(f"Found {n_long} segments > 30 sec in {manifest}")

        self.processor = processor
        self.clap_embeddings = None
        self.clap_id_to_idx = None
        self.clap_meta = None
        if clap_emb_path is not None:
            self.clap_embeddings, self.clap_id_to_idx, self.clap_meta = load_clap_embedding_store(clap_emb_path)
            missing_ids = [
                sid for sid in self.df["sample_id"].astype(str).tolist()
                if sid not in self.clap_id_to_idx
            ]
            if missing_ids:
                raise ValueError(
                    f"{len(missing_ids)} samples in manifest are missing CLAP embeddings. "
                    f"First: {missing_ids[:5]}"
                )

        print(
            f"{manifest}\n"
            f"  samples = {len(self.df)}\n"
            f"  hours   = {self.df['duration'].astype(float).sum()/3600:.2f}\n"
            f"  CLAP    = {'yes' if self.clap_embeddings is not None else 'no'}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        sid = str(r["sample_id"])
        wav = load_segment(r["audio_path"], r["start"], r["end"])

        input_features = self.processor.feature_extractor(
            wav,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            str(r["text_spoken"]),
            return_tensors="pt",
        ).input_ids[0]

        item = {
            "input_features": input_features,
            "labels": labels,
            "text": str(r["text_spoken"]),
            "sample_id": sid,
        }
        if self.clap_embeddings is not None:
            item["clap_emb"] = self.clap_embeddings[self.clap_id_to_idx[sid]]
        return item


class DataCollatorSpeechSeq2SeqWithClap:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, batch):
        input_features = torch.stack([b["input_features"] for b in batch], dim=0)
        label_features = [{"input_ids": b["labels"]} for b in batch]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

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
    """Exact post-encoder gated adapter structure from the team's v2 code."""
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
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, h):
        x = self.norm(h).transpose(1, 2)
        x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x).transpose(1, 2)
        return h + self.scale * x


def build_adapter(adapter_type, hidden_dim, bottleneck_dim, dropout, scale_init):
    t = adapter_type.lower()
    if t == "none":
        return IdentityAdapter()
    if t == "linear_residual":
        return LinearResidualAdapter(hidden_dim, scale_init)
    if t == "residual_mlp":
        return ResidualMLPAdapter(hidden_dim, bottleneck_dim, dropout, scale_init)
    if t == "bottleneck":
        return BottleneckAdapter(hidden_dim, bottleneck_dim, dropout, scale_init)
    if t == "gated":
        return GatedAdapter(hidden_dim, bottleneck_dim, dropout, scale_init)
    if t == "conv1d":
        return Conv1DAdapter(hidden_dim, 3, dropout, scale_init)
    raise ValueError(f"Unknown adapter_type: {adapter_type}")


class MeanPooler(nn.Module):
    def forward(self, h):
        # Intentionally matches the original MathSpeech implementation:
        # plain mean over Whisper encoder time steps, without a padding mask.
        return h.mean(dim=1)


class CLSPooler(nn.Module):
    def forward(self, h):
        return h[:, 0]


class AttentionPooler(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, h):
        attn = torch.softmax(self.score(h).squeeze(-1), dim=-1)
        return torch.sum(h * attn.unsqueeze(-1), dim=1)


def build_pooler(pool_type, hidden_dim):
    t = pool_type.lower()
    if t == "mean":
        return MeanPooler()
    if t == "cls":
        return CLSPooler()
    if t == "attn":
        return AttentionPooler(hidden_dim)
    raise ValueError(f"Unknown pool_type: {pool_type}")


def cosine_alignment_loss(z, target):
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - F.cosine_similarity(z, target, dim=-1).mean()


def mse_alignment_loss(z, target):
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(z, target)


def clip_contrastive_loss(z, target, temperature=0.07):
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (z @ target.t()) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )


def compute_alignment_loss(
    z, target, loss_type="cosine", temperature=0.07,
    lambda_cosine=1.0, lambda_mse=1.0, lambda_clip=1.0,
):
    loss_type = loss_type.lower()
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
        raise ValueError(f"Unknown align_loss_type: {loss_type}")
    return total, {"cosine": cos.detach(), "mse": mse.detach(), "clip": clip.detach()}


class WhisperSemanticASR(nn.Module):
    def __init__(
        self,
        whisper_name="openai/whisper-base",
        clap_dim=512,
        adapter_type="gated",
        pool_type="mean",
        adapter_bottleneck=256,
        dropout=0.1,
        adapter_scale_init=0.01,
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

        # Ours always freezes Whisper.
        for p in self.whisper.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True
        for p in self.pooler.parameters():
            p.requires_grad = True
        for p in self.align_head.parameters():
            p.requires_grad = True

    def encode_with_adapter(self, input_features, return_original=False):
        enc = self.whisper.model.encoder(input_features)
        h_original = enc.last_hidden_state
        h_adapted = self.adapter(h_original)
        if return_original:
            return h_adapted, h_original
        return h_adapted

    def forward(
        self, input_features, labels, clap_emb=None,
        lambda_align=0.05, lambda_hidden=0.1,
        align_loss_type="cosine", temperature=0.07,
        lambda_cosine=1.0, lambda_mse=1.0, lambda_clip=1.0,
    ):
        h_adapted, h_original = self.encode_with_adapter(input_features, True)
        enc_out = BaseModelOutput(last_hidden_state=h_adapted)
        outputs = self.whisper(
            encoder_outputs=enc_out,
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
                z, clap_emb, align_loss_type, temperature,
                lambda_cosine, lambda_mse, lambda_clip,
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
        }

    @torch.no_grad()
    def generate(self, input_features, **kwargs):
        h_adapted = self.encode_with_adapter(input_features)
        enc_out = BaseModelOutput(last_hidden_state=h_adapted)
        return self.whisper.generate(encoder_outputs=enc_out, **kwargs)


def normalize_text(processor, text):
    tok = processor.tokenizer
    text = str(text).strip()
    if hasattr(tok, "normalize"):
        return tok.normalize(text).strip()
    if hasattr(tok, "_normalize"):
        return tok._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found.")


def save_checkpoint(path, model, args, clap_dim, epoch, dev_metrics):
    payload = {
        "adapter_state_dict": model.adapter.state_dict(),
        "pooler_state_dict": model.pooler.state_dict(),
        "align_head_state_dict": model.align_head.state_dict(),
        "args": vars(args),
        "clap_dim": clap_dim,
        "epoch": epoch,
        "dev_metrics": dev_metrics,
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate_dev(model, loader, processor, device, args, use_amp):
    """CLAP-free dev evaluation used only for ASR checkpoint selection."""
    model.eval()
    refs, hyps = [], []

    for batch in tqdm(loader, desc="dev"):
        input_features = batch["input_features"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_ids = model.generate(
                input_features=input_features,
                num_beams=args.eval_num_beams,
                max_new_tokens=args.generation_max_new_tokens,
                do_sample=False,
            )

        pred_text = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        for ref, hyp in zip(batch["texts"], pred_text):
            r = normalize_text(processor, ref)
            h = normalize_text(processor, hyp)
            if r:
                refs.append(r)
                hyps.append(h)

    return {
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "num_eval_utterances": len(refs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--train_clap_emb", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--whisper_name", default="openai/whisper-base")

    # Final MathSpeech configuration from team code / slides.
    ap.add_argument("--adapter_type", default="gated",
                    choices=["none", "linear_residual", "residual_mlp", "bottleneck", "gated", "conv1d"])
    ap.add_argument("--pool_type", default="mean", choices=["mean", "cls", "attn"])
    ap.add_argument("--adapter_bottleneck", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--adapter_scale_init", type=float, default=0.01)

    ap.add_argument("--align_loss_type", default="cosine",
                    choices=["none", "cosine", "mse", "clip", "cosine_clip", "cosine_mse", "all"])
    ap.add_argument("--lambda_align", type=float, default=0.05)
    ap.add_argument("--lambda_hidden", type=float, default=0.1)
    ap.add_argument("--lambda_cosine", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true",
                    help="Optional; original MathSpeech projector code trained in fp32.")

    ap.add_argument("--eval_num_beams", type=int, default=5)
    ap.add_argument("--generation_max_new_tokens", type=int, default=256)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_dev_samples", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")
    print("device:", device)
    print("fp16:", use_amp)

    processor = WhisperProcessor.from_pretrained(
        args.whisper_name, language="English", task="transcribe"
    )

    train_ds = M3AVSemanticDataset(
        args.train, processor, args.train_clap_emb, args.max_train_samples
    )
    dev_ds = M3AVSemanticDataset(
        args.dev, processor, None, args.max_dev_samples
    )

    # Lecture-CLAP teacher embeddings are used only for Stage-2 training.
    # Dev/test remain CLAP-free and are used only for ASR evaluation/checkpoint selection.
    train_meta = train_ds.clap_meta
    clap_dim = int(train_meta["embedding_dim"])
    model = WhisperSemanticASR(
        whisper_name=args.whisper_name,
        clap_dim=clap_dim,
        adapter_type=args.adapter_type,
        pool_type=args.pool_type,
        adapter_bottleneck=args.adapter_bottleneck,
        dropout=args.dropout,
        adapter_scale_init=args.adapter_scale_init,
    ).to(device)

    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_ids
    model.whisper.generation_config.forced_decoder_ids = forced_ids

    collator = DataCollatorSpeechSeq2SeqWithClap(
        processor, model.whisper.config.decoder_start_token_id
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers, pin_memory=True,
    )

    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    n_whisper = sum(p.numel() for p in model.whisper.parameters())

    print("=" * 72)
    print("M3AV OURS CONFIGURATION")
    print("=" * 72)
    print("whisper              :", args.whisper_name)
    print("adapter              :", args.adapter_type)
    print("pool                 :", args.pool_type)
    print("align loss           :", args.align_loss_type)
    print("lambda_align         :", args.lambda_align)
    print("lambda_hidden        :", args.lambda_hidden)
    print("adapter_scale_init   :", args.adapter_scale_init)
    print("lr                   :", args.lr)
    print("epochs               :", args.epochs)
    print("CLAP teacher         :", train_meta.get("ckpt_path"))
    print("CLAP text template   :", train_meta.get("text_template"))
    print(f"trainable params     : {n_trainable:,} ({n_trainable/1e6:.6f}M)")
    print(f"Whisper base params  : {n_whisper:,} ({n_whisper/1e6:.6f}M)")
    print(f"trainable/base ratio : {100*n_trainable/n_whisper:.4f}%")
    print(f"all instantiated     : {n_total:,}")
    print("=" * 72)

    if args.adapter_type == "gated" and args.pool_type == "mean" and clap_dim == 512:
        expected = 790_273
        if n_trainable != expected:
            raise RuntimeError(
                f"Expected {expected:,} trainable params for final Gated+Mean configuration, "
                f"but found {n_trainable:,}."
            )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dev_wer = float("inf")
    best_epoch = None
    log_rows = []
    log_path = save_dir / "train_log.csv"
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "hidden": 0.0, "align": 0.0,
                "cosine": 0.0, "mse": 0.0, "clip": 0.0}
        align_count = 0
        optimizer_steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for step, batch in enumerate(pbar, start=1):
            input_features = batch["input_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            clap_emb = batch["clap_emb"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
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
                loss = out["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            sums["loss"] += out["loss"].item()
            sums["ce"] += out["ce_loss"].item()
            sums["hidden"] += out["hidden_loss"].item()
            sums["cosine"] += out["cosine_loss"].item()
            sums["mse"] += out["mse_loss"].item()
            sums["clip"] += out["clip_loss"].item()
            if out["align_loss"] is not None:
                sums["align"] += out["align_loss"].item()
                align_count += 1

            pbar.set_postfix(
                loss=f"{out['loss'].item():.4f}",
                ce=f"{out['ce_loss'].item():.4f}",
                hid=f"{out['hidden_loss'].item():.4f}",
                align=(f"{out['align_loss'].item():.4f}" if out["align_loss"] is not None else "0"),
            )

        denom = max(len(train_loader), 1)
        train_stats = {
            "loss": sums["loss"] / denom,
            "ce": sums["ce"] / denom,
            "hidden": sums["hidden"] / denom,
            "align": sums["align"] / max(align_count, 1),
            "cosine": sums["cosine"] / denom,
            "mse": sums["mse"] / denom,
            "clip": sums["clip"] / denom,
            "optimizer_steps": optimizer_steps,
        }

        dev_stats = evaluate_dev(model, dev_loader, processor, device, args, use_amp)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_ce={train_stats['ce']:.4f} "
            f"train_hidden={train_stats['hidden']:.4f} "
            f"train_align={train_stats['align']:.4f} | "
            f"dev_WER={dev_stats['wer']:.6f} "
            f"dev_CER={dev_stats['cer']:.6f}"
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"dev_{k}": v for k, v in dev_stats.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        save_checkpoint(save_dir / "last.pt", model, args, clap_dim, epoch, dev_stats)
        if dev_stats["wer"] < best_dev_wer:
            best_dev_wer = dev_stats["wer"]
            best_epoch = epoch
            save_checkpoint(save_dir / "best.pt", model, args, clap_dim, epoch, dev_stats)
            print(f"saved best.pt (dev WER={best_dev_wer:.6f})")

    summary = {
        "best_epoch": best_epoch,
        "best_dev_wer": best_dev_wer,
        "trainable_params": n_trainable,
        "whisper_base_params": n_whisper,
        "trainable_base_ratio_percent": 100 * n_trainable / n_whisper,
        "clap_teacher": train_meta.get("ckpt_path"),
        "clap_text_template": train_meta.get("text_template"),
        "args": vars(args),
    }
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("TRAINING COMPLETE")
    print("best epoch   :", best_epoch)
    print("best dev WER :", best_dev_wer)
    print("best ckpt    :", save_dir / "best.pt")
    print("=" * 72)


if __name__ == "__main__":
    main()
