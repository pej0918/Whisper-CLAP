import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import whisper as openai_whisper

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


# =========================================================
# Dataset
# =========================================================
class MathSpeechDataset(Dataset):
    def __init__(self, df, indices, audio_dir, processor, clap_embs=None):
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.audio_dir = audio_dir
        self.processor = processor
        self.clap_embs = clap_embs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        text = str(row["transcription"])
        audio_path = os.path.join(self.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        audio = openai_whisper.load_audio(audio_path)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
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
            "real_idx": real_idx,
        }

        if self.clap_embs is not None:
            item["clap_emb"] = self.clap_embs[real_idx]

        return item


class DataCollatorSpeechSeq2SeqWithClap:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = torch.stack([b["input_features"] for b in batch], dim=0)

        label_features = [{"input_ids": b["labels"]} for b in batch]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        decoder_start_token_id = self.processor.tokenizer.bos_token_id
        if labels.shape[1] > 0 and (labels[:, 0] == decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        output = {
            "input_features": input_features,
            "labels": labels,
            "texts": [b["text"] for b in batch],
            "real_indices": [b["real_idx"] for b in batch],
        }

        if "clap_emb" in batch[0]:
            output["clap_emb"] = torch.stack([b["clap_emb"] for b in batch], dim=0)

        return output


# =========================================================
# Projector / Adapter variants
# =========================================================
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
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, h):
        x = self.norm(h)          # [B, T, D]
        x = x.transpose(1, 2)     # [B, D, T]
        x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)     # [B, T, D]
        return h + self.scale * x


def build_adapter(adapter_type, hidden_dim, bottleneck_dim, dropout, scale_init=0.01):
    adapter_type = adapter_type.lower()

    if adapter_type == "none":
        return IdentityAdapter()
    if adapter_type == "linear_residual":
        return LinearResidualAdapter(hidden_dim, scale_init=scale_init)
    if adapter_type == "residual_mlp":
        return ResidualMLPAdapter(
            hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            scale_init=scale_init,
        )
    if adapter_type == "bottleneck":
        return BottleneckAdapter(
            hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            scale_init=scale_init,
        )
    if adapter_type == "gated":
        return GatedAdapter(
            hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            scale_init=scale_init,
        )
    if adapter_type == "conv1d":
        return Conv1DAdapter(
            hidden_dim,
            kernel_size=3,
            dropout=dropout,
            scale_init=scale_init,
        )

    raise ValueError(f"Unknown adapter_type: {adapter_type}")


# =========================================================
# Pooling variants
# =========================================================
class MeanPooler(nn.Module):
    def forward(self, h):
        return h.mean(dim=1)


class CLSPooler(nn.Module):
    def forward(self, h):
        return h[:, 0]


class AttentionPooler(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        attn = self.score(h).squeeze(-1)  # [B, T]
        attn = torch.softmax(attn, dim=-1)
        pooled = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return pooled


def build_pooler(pool_type, hidden_dim):
    pool_type = pool_type.lower()

    if pool_type == "mean":
        return MeanPooler()
    if pool_type == "cls":
        return CLSPooler()
    if pool_type == "attn":
        return AttentionPooler(hidden_dim)

    raise ValueError(f"Unknown pool_type: {pool_type}")


# =========================================================
# Loss variants
# =========================================================
def cosine_alignment_loss(z, target):
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - F.cosine_similarity(z, target, dim=-1).mean()


def mse_alignment_loss(z, target):
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(z, target)


def clip_contrastive_loss(z, target, temperature=0.07):
    """
    In-batch InfoNCE.
    z[i] should match target[i].
    """
    z = F.normalize(z, dim=-1)
    target = F.normalize(target, dim=-1)

    logits = z @ target.t()
    logits = logits / temperature

    labels = torch.arange(z.size(0), device=z.device)

    loss_z_to_t = F.cross_entropy(logits, labels)
    loss_t_to_z = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_z_to_t + loss_t_to_z)


def compute_alignment_loss(
    z,
    target,
    loss_type="cosine",
    temperature=0.07,
    lambda_cosine=1.0,
    lambda_mse=1.0,
    lambda_clip=1.0,
):
    loss_type = loss_type.lower()

    zero = z.new_tensor(0.0)

    if loss_type == "none":
        return zero, {
            "cosine": zero.detach(),
            "mse": zero.detach(),
            "clip": zero.detach(),
        }

    cos = cosine_alignment_loss(z, target)
    mse = mse_alignment_loss(z, target)
    clip = clip_contrastive_loss(z, target, temperature=temperature)

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

    return total, {
        "cosine": cos.detach(),
        "mse": mse.detach(),
        "clip": clip.detach(),
    }


# =========================================================
# Model
# =========================================================
class WhisperSemanticASR(nn.Module):
    def __init__(
        self,
        whisper_name="openai/whisper-base",
        clap_dim=512,
        adapter_type="residual_mlp",
        pool_type="mean",
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

        # Keep adapted encoder states close to original Whisper encoder space.
        # This stabilizes decoder EOS/stopping behavior.
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


# =========================================================
# Utils
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split(n, seed=42, train_ratio=0.8, valid_ratio=0.1):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    return train_idx, valid_idx, test_idx


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_hidden = 0.0
    total_align = 0.0
    total_cos = 0.0
    total_mse = 0.0
    total_clip = 0.0
    align_count = 0

    for batch in tqdm(loader, desc="valid"):
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

        total_loss += out["loss"].item()
        total_ce += out["ce_loss"].item()
        total_hidden += out["hidden_loss"].item()
        total_cos += out["cosine_loss"].item()
        total_mse += out["mse_loss"].item()
        total_clip += out["clip_loss"].item()

        if out["align_loss"] is not None:
            total_align += out["align_loss"].item()
            align_count += 1

    denom = max(len(loader), 1)

    return {
        "loss": total_loss / denom,
        "ce": total_ce / denom,
        "hidden": total_hidden / denom,
        "align": total_align / max(align_count, 1),
        "cosine": total_cos / denom,
        "mse": total_mse / denom,
        "clip": total_clip / denom,
    }


# =========================================================
# Train
# =========================================================
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
    print("whisper model:", args.whisper_name)
    print("adapter_type:", args.adapter_type)
    print("pool_type:", args.pool_type)
    print("align_loss_type:", args.align_loss_type)
    print("lambda_align:", args.lambda_align)
    print("lambda_hidden:", args.lambda_hidden)
    print("adapter_scale_init:", args.adapter_scale_init)
    print("lr:", args.lr)

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

    print(f"num samples: {n}")
    print(f"split sizes: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")
    print("saved split to:", split_path)

    processor = WhisperProcessor.from_pretrained(
        args.whisper_name,
        language="en",
        task="transcribe",
    )

    clap_embs = None
    clap_dim = 512

    use_align = args.lambda_align > 0 and args.align_loss_type != "none"

    if use_align:
        if not os.path.exists(args.clap_emb_path):
            raise FileNotFoundError(args.clap_emb_path)

        clap_embs = torch.load(args.clap_emb_path, map_location="cpu").float()
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = WhisperSemanticASR(
        whisper_name=args.whisper_name,
        clap_dim=clap_dim,
        adapter_type=args.adapter_type,
        pool_type=args.pool_type,
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
    print("freeze_whisper:", args.freeze_whisper)
    print("clap_dim:", clap_dim)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_valid_loss = float("inf")

    log_path = os.path.join(args.save_dir, "train_log.csv")
    log_rows = []

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
            f"train_align={train_stats['align']:.4f}, "
            f"train_cos={train_stats['cosine']:.4f}, "
            f"train_mse={train_stats['mse']:.4f}, "
            f"train_clip={train_stats['clip']:.4f} | "
            f"valid_loss={valid_stats['loss']:.4f}, "
            f"valid_ce={valid_stats['ce']:.4f}, "
            f"valid_hidden={valid_stats['hidden']:.4f}, "
            f"valid_align={valid_stats['align']:.4f}, "
            f"valid_cos={valid_stats['cosine']:.4f}, "
            f"valid_mse={valid_stats['mse']:.4f}, "
            f"valid_clip={valid_stats['clip']:.4f}"
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

        ckpt_common = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "clap_dim": clap_dim,
            "epoch": epoch,
            "train_stats": train_stats,
            "valid_stats": valid_stats,
        }

        torch.save(ckpt_common, os.path.join(args.save_dir, "last.pt"))

        if valid_stats["loss"] < best_valid_loss:
            best_valid_loss = valid_stats["loss"]
            ckpt_common["best_valid_loss"] = best_valid_loss
            torch.save(ckpt_common, os.path.join(args.save_dir, "best.pt"))
            print("saved best:", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
