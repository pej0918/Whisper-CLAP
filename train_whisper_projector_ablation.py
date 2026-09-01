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
from torch.nn.utils.rnn import pad_sequence

import whisper as openai_whisper
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_or_load_split(num_samples, save_dir, split_path=None, seed=42):
    if split_path is not None and os.path.exists(split_path):
        return torch.load(split_path, map_location="cpu", weights_only=False)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "split_indices.pt")

    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    n_train = int(num_samples * 0.8)
    n_valid = int(num_samples * 0.1)

    split = {
        "train_idx": indices[:n_train].tolist(),
        "valid_idx": indices[n_train:n_train + n_valid].tolist(),
        "test_idx": indices[n_train + n_valid:].tolist(),
    }

    torch.save(split, path)
    print("saved split to:", path)
    return split


class MathSpeechProjectorDataset(Dataset):
    def __init__(
        self,
        df,
        indices,
        audio_dir,
        processor,
        clap_emb=None,
        text_col="transcription",
    ):
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.audio_dir = audio_dir
        self.processor = processor
        self.clap_emb = clap_emb
        self.text_col = text_col

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        real_idx = self.indices[item]
        audio_path = os.path.join(self.audio_dir, f"{real_idx + 1}.mp3")

        audio = openai_whisper.load_audio(audio_path)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        text = str(self.df[self.text_col].iloc[real_idx]).strip()

        labels = self.processor.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids[0]

        if self.clap_emb is None:
            target_emb = torch.zeros(1)
        else:
            target_emb = self.clap_emb[real_idx].float()

        return {
            "input_features": input_features,
            "labels": labels,
            "target_emb": target_emb,
            "real_idx": real_idx,
        }


class ProjectorCollator:
    def __init__(self, processor, use_clap):
        self.processor = processor
        self.use_clap = use_clap

    def __call__(self, batch):
        input_features = torch.stack([x["input_features"] for x in batch], dim=0)

        labels = [x["labels"] for x in batch]
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        real_idx = torch.tensor([x["real_idx"] for x in batch], dtype=torch.long)

        if self.use_clap:
            target_emb = torch.stack([x["target_emb"] for x in batch], dim=0)
        else:
            target_emb = None

        return {
            "input_features": input_features,
            "labels": labels,
            "target_emb": target_emb,
            "real_idx": real_idx,
        }


class ResidualMLPAdapter(nn.Module):
    def __init__(self, dim, bottleneck=256, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x):
        return x + self.scale * self.net(x)


class BottleneckAdapter(nn.Module):
    def __init__(self, dim, bottleneck=128, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x):
        h = self.norm(x)
        h = self.down(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + self.scale * h


class GatedAdapter(nn.Module):
    def __init__(self, dim, bottleneck=256, dropout=0.1, scale_init=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x):
        h = self.norm(x)
        v = self.value(h)
        g = self.gate(h)
        return x + self.scale * g * v


class LinearResidualAdapter(nn.Module):
    def __init__(self, dim, scale_init=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x):
        return x + self.scale * self.proj(self.norm(x))


class IdentityAdapter(nn.Module):
    def forward(self, x):
        return x


def build_adapter(adapter_type, dim, bottleneck, dropout, scale_init):
    if adapter_type == "residual_mlp":
        return ResidualMLPAdapter(dim, bottleneck, dropout, scale_init)
    if adapter_type == "gated":
        return GatedAdapter(dim, bottleneck, dropout, scale_init)
    if adapter_type == "bottleneck":
        return BottleneckAdapter(dim, bottleneck, dropout, scale_init)
    if adapter_type == "linear_residual":
        return LinearResidualAdapter(dim, scale_init)
    if adapter_type == "none":
        return IdentityAdapter()
    raise ValueError(f"Unknown adapter_type: {adapter_type}")


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        score = self.score(x).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return pooled


class WhisperSemanticASR(nn.Module):
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
        self.whisper.config.forced_decoder_ids = None
        self.whisper.generation_config.forced_decoder_ids = None
        self.whisper.config.suppress_tokens = []
        self.whisper.generation_config.suppress_tokens = []

        self.d_model = self.whisper.config.d_model
        self.clap_dim = clap_dim

        self.adapter_type = adapter_type
        self.pool_type = pool_type
        self.adapter_position = adapter_position
        self.encoder_layer = encoder_layer

        self.post_adapter = build_adapter(
            adapter_type=adapter_type,
            dim=self.d_model,
            bottleneck=adapter_bottleneck,
            dropout=dropout,
            scale_init=adapter_scale_init,
        )

        self.mid_adapter = build_adapter(
            adapter_type=adapter_type,
            dim=self.d_model,
            bottleneck=adapter_bottleneck,
            dropout=dropout,
            scale_init=adapter_scale_init,
        )

        if pool_type == "attn":
            self.pool = AttentionPool(self.d_model)
        elif pool_type in ["mean", "cls"]:
            self.pool = None
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.align_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, clap_dim),
        )

        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad = False

    def _pool_hidden(self, hidden):
        if self.pool_type == "mean":
            return hidden.mean(dim=1)
        if self.pool_type == "cls":
            return hidden[:, 0]
        if self.pool_type == "attn":
            return self.pool(hidden)
        raise ValueError(self.pool_type)

    def _run_encoder_with_optional_mid_adapter(self, input_features):
        handle = None

        use_mid = self.adapter_position in ["encoder_layer", "both"]

        if use_mid:
            layers = self.whisper.model.encoder.layers
            layer_idx = min(max(self.encoder_layer, 0), len(layers) - 1)

            def hook_fn(module, inputs, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h = self.mid_adapter(h)
                    return (h,) + output[1:]
                return self.mid_adapter(output)

            handle = layers[layer_idx].register_forward_hook(hook_fn)

        enc = self.whisper.model.encoder(
            input_features,
            return_dict=True,
        )

        if handle is not None:
            handle.remove()

        return enc.last_hidden_state

    def encode_adapted(self, input_features, return_original=False):
        original_hidden = None

        if return_original:
            with torch.no_grad():
                original_hidden = self.whisper.model.encoder(
                    input_features,
                    return_dict=True,
                ).last_hidden_state.detach()

        hidden = self._run_encoder_with_optional_mid_adapter(input_features)

        if self.adapter_position in ["post_encoder", "both"]:
            hidden = self.post_adapter(hidden)

        return hidden, original_hidden

    def get_alignment_embedding(self, input_features):
        hidden, _ = self.encode_adapted(input_features, return_original=False)
        pooled = self._pool_hidden(hidden)
        emb = self.align_head(pooled)
        return emb

    def forward(
        self,
        input_features,
        labels=None,
        target_emb=None,
        align_loss_type="none",
        lambda_align=0.0,
        lambda_cosine=1.0,
        lambda_clip=1.0,
        lambda_hidden=0.1,
        temperature=0.07,
    ):
        need_original = lambda_hidden > 0.0
        hidden, original_hidden = self.encode_adapted(
            input_features,
            return_original=need_original,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=hidden)

        out = self.whisper(
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=True,
        )

        ce_loss = out.loss
        total_loss = ce_loss

        align_loss = torch.zeros([], device=ce_loss.device)
        cosine_loss = torch.zeros([], device=ce_loss.device)
        clip_loss = torch.zeros([], device=ce_loss.device)
        hidden_loss = torch.zeros([], device=ce_loss.device)

        pooled = self._pool_hidden(hidden)
        pred_emb = self.align_head(pooled)

        if align_loss_type != "none":
            if target_emb is None:
                raise ValueError("target_emb is required when align_loss_type != none")

            pred_norm = F.normalize(pred_emb.float(), dim=-1)
            target_norm = F.normalize(target_emb.float(), dim=-1)

            cosine_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

            logits = pred_norm @ target_norm.t()
            logits = logits / temperature
            labels_clip = torch.arange(logits.size(0), device=logits.device)
            clip_loss_a = F.cross_entropy(logits, labels_clip)
            clip_loss_b = F.cross_entropy(logits.t(), labels_clip)
            clip_loss = 0.5 * (clip_loss_a + clip_loss_b)

            if align_loss_type == "cosine":
                align_loss = cosine_loss
            elif align_loss_type == "clip":
                align_loss = clip_loss
            elif align_loss_type == "cosine_clip":
                align_loss = lambda_cosine * cosine_loss + lambda_clip * clip_loss
            elif align_loss_type == "mse":
                align_loss = F.mse_loss(pred_norm, target_norm)
            else:
                raise ValueError(f"Unknown align_loss_type: {align_loss_type}")

            total_loss = total_loss + lambda_align * align_loss

        if lambda_hidden > 0.0 and original_hidden is not None:
            hidden_loss = F.mse_loss(hidden.float(), original_hidden.float())
            total_loss = total_loss + lambda_hidden * hidden_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "align_loss": align_loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "clip_loss": clip_loss.detach(),
            "hidden_loss": hidden_loss.detach(),
            "logits": out.logits,
        }

    @torch.no_grad()
    def generate(self, input_features, **generate_kwargs):
        hidden, _ = self.encode_adapted(input_features, return_original=False)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden)

        return self.whisper.generate(
            encoder_outputs=encoder_outputs,
            **generate_kwargs,
        )


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args):
    model.train()

    total = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "align_loss": 0.0,
        "hidden_loss": 0.0,
    }
    n = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train")):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        target_emb = batch["target_emb"]
        if target_emb is not None:
            target_emb = target_emb.to(device)

        with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
            out = model(
                input_features=input_features,
                labels=labels,
                target_emb=target_emb,
                align_loss_type=args.align_loss_type,
                lambda_align=args.lambda_align,
                lambda_cosine=args.lambda_cosine,
                lambda_clip=args.lambda_clip,
                lambda_hidden=args.lambda_hidden,
                temperature=args.temperature,
            )
            loss = out["loss"] / args.grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        total["loss"] += out["loss"].item()
        total["ce_loss"] += out["ce_loss"].item()
        total["align_loss"] += out["align_loss"].item()
        total["hidden_loss"] += out["hidden_loss"].item()
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def evaluate_loss(model, loader, device, args):
    model.eval()

    total = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "align_loss": 0.0,
        "hidden_loss": 0.0,
    }
    n = 0

    for batch in tqdm(loader, desc="valid"):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        target_emb = batch["target_emb"]
        if target_emb is not None:
            target_emb = target_emb.to(device)

        with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
            out = model(
                input_features=input_features,
                labels=labels,
                target_emb=target_emb,
                align_loss_type=args.align_loss_type,
                lambda_align=args.lambda_align,
                lambda_cosine=args.lambda_cosine,
                lambda_clip=args.lambda_clip,
                lambda_hidden=args.lambda_hidden,
                temperature=args.temperature,
            )

        total["loss"] += out["loss"].item()
        total["ce_loss"] += out["ce_loss"].item()
        total["align_loss"] += out["align_loss"].item()
        total["hidden_loss"] += out["hidden_loss"].item()
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--clap_emb_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split_path", type=str, default=None)

    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--text_col", type=str, default="transcription")

    parser.add_argument("--adapter_type", type=str, default="residual_mlp",
                        choices=["residual_mlp", "gated", "bottleneck", "linear_residual", "none"])
    parser.add_argument("--pool_type", type=str, default="mean",
                        choices=["mean", "attn", "cls"])
    parser.add_argument("--adapter_position", type=str, default="post_encoder",
                        choices=["post_encoder", "encoder_layer", "both"])
    parser.add_argument("--encoder_layer", type=int, default=6)
    parser.add_argument("--adapter_bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter_scale_init", type=float, default=0.01)

    parser.add_argument("--align_loss_type", type=str, default="none",
                        choices=["none", "cosine", "clip", "cosine_clip", "mse"])
    parser.add_argument("--lambda_align", type=float, default=0.0)
    parser.add_argument("--lambda_cosine", type=float, default=1.0)
    parser.add_argument("--lambda_clip", type=float, default=0.1)
    parser.add_argument("--lambda_hidden", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--freeze_whisper", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)
    split = make_or_load_split(
        num_samples=len(df),
        save_dir=args.save_dir,
        split_path=args.split_path,
        seed=args.seed,
    )

    processor = WhisperProcessor.from_pretrained(
        args.whisper_name,
        language="en",
        task="transcribe",
    )

    clap_emb = None
    clap_dim = 512

    if args.clap_emb_path is not None and os.path.exists(args.clap_emb_path):
        clap_emb = torch.load(args.clap_emb_path, map_location="cpu", weights_only=False)
        if isinstance(clap_emb, dict):
            if "embeddings" in clap_emb:
                clap_emb = clap_emb["embeddings"]
            elif "text_emb" in clap_emb:
                clap_emb = clap_emb["text_emb"]
            else:
                raise ValueError(f"Unknown clap embedding keys: {clap_emb.keys()}")
        clap_emb = clap_emb.float()
        clap_dim = clap_emb.shape[-1]
        print("loaded clap_emb:", clap_emb.shape)

    if args.align_loss_type != "none" and clap_emb is None:
        raise ValueError("--clap_emb_path is required when align_loss_type != none")

    model = WhisperSemanticASR(
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

    train_dataset = MathSpeechProjectorDataset(
        df=df,
        indices=split["train_idx"],
        audio_dir=args.audio_dir,
        processor=processor,
        clap_emb=clap_emb,
        text_col=args.text_col,
    )

    valid_dataset = MathSpeechProjectorDataset(
        df=df,
        indices=split["valid_idx"],
        audio_dir=args.audio_dir,
        processor=processor,
        clap_emb=clap_emb,
        text_col=args.text_col,
    )

    use_clap = args.align_loss_type != "none"

    collator = ProjectorCollator(
        processor=processor,
        use_clap=use_clap,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())

    best_valid = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_log = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
        )

        valid_log = evaluate_loss(
            model=model,
            loader=valid_loader,
            device=device,
            args=args,
        )

        print("train:", train_log)
        print("valid:", valid_log)

        torch.save(
            {
                "epoch": epoch,
                "args": vars(args),
                "model_state_dict": model.state_dict(),
                "valid_loss": valid_log["loss"],
                "clap_dim": clap_dim,
            },
            os.path.join(args.save_dir, "last.pt"),
        )

        if valid_log["loss"] < best_valid:
            best_valid = valid_log["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "args": vars(args),
                    "model_state_dict": model.state_dict(),
                    "valid_loss": valid_log["loss"],
                    "clap_dim": clap_dim,
                },
                os.path.join(args.save_dir, "best.pt"),
            )
            print("saved best")

    print("best_valid_loss:", best_valid)


if __name__ == "__main__":
    main()