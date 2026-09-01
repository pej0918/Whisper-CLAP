import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from jiwer import wer as jiwer_wer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

from asr_dataset_utils import ASRCollator, ASRDataset, add_dataset_args, load_records_from_args
from compute_asr_metrics import normalize_text
from mathspeech_utils import set_seed


class KAUSTStyleResidualAdapter(nn.Module):
    """Residual bottleneck adapter used after an encoder block.

    This mirrors the adapter placement/shape used by KAUST-Whisper-Adapter:
    down-project hidden states, apply GELU, up-project, then add the result
    residually to the block output. We implement it inside the Hugging Face
    Whisper stack via encoder-layer forward hooks instead of modifying the
    OpenAI Whisper source tree directly.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, h):
        return h + self.up(self.dropout(self.act(self.down(h))))


class SingleOutputResidualAdapter(nn.Module):
    """Backward-compatible single encoder-output adapter used by older runs."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256, dropout: float = 0.1, scale_init: float = 0.01):
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


class ResidualAdapterWhisper(nn.Module):
    def __init__(
        self,
        whisper_name: str = "openai/whisper-base",
        adapter_bottleneck: int = 256,
        dropout: float = 0.0,
        adapter_scale_init: float = 0.01,
        freeze_whisper: bool = True,
        adapter_style: str = "kaust_layerwise",
    ):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        self.whisper.config.use_cache = False
        self.adapter_style = adapter_style
        hidden_dim = self.whisper.config.d_model
        self._adapter_hooks = []

        if self.adapter_style == "kaust_layerwise":
            encoder_layers = self.whisper.model.encoder.layers
            self.adapters = nn.ModuleList(
                [
                    KAUSTStyleResidualAdapter(
                        hidden_dim=hidden_dim,
                        bottleneck_dim=adapter_bottleneck,
                        dropout=dropout,
                    )
                    for _ in range(len(encoder_layers))
                ]
            )
            for layer_idx, layer in enumerate(encoder_layers):
                self._adapter_hooks.append(layer.register_forward_hook(self._make_layerwise_adapter_hook(layer_idx)))
        elif self.adapter_style == "single_encoder_output":
            self.adapter = SingleOutputResidualAdapter(hidden_dim, adapter_bottleneck, dropout, adapter_scale_init)
        else:
            raise ValueError(f"Unknown adapter_style: {self.adapter_style}")

        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad = False

        if self.adapter_style == "kaust_layerwise":
            for p in self.adapters.parameters():
                p.requires_grad = True
        else:
            for p in self.adapter.parameters():
                p.requires_grad = True

    def _make_layerwise_adapter_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            adapter = self.adapters[layer_idx]
            if isinstance(output, tuple):
                hidden = adapter(output[0])
                return (hidden,) + output[1:]
            return adapter(output)

        return hook

    def adapter_state_dict(self):
        if self.adapter_style == "kaust_layerwise":
            return self.adapters.state_dict()
        return self.adapter.state_dict()

    def load_adapter_state_dict(self, state_dict, strict: bool = True):
        if self.adapter_style == "kaust_layerwise":
            return self.adapters.load_state_dict(state_dict, strict=strict)
        return self.adapter.load_state_dict(state_dict, strict=strict)

    def encode_with_adapter(self, input_features):
        enc = self.whisper.model.encoder(input_features, return_dict=True)
        h = enc.last_hidden_state
        if self.adapter_style == "single_encoder_output":
            h = self.adapter(h)
        return h

    def forward(self, input_features, labels):
        h = self.encode_with_adapter(input_features)
        out = self.whisper(
            encoder_outputs=BaseModelOutput(last_hidden_state=h),
            labels=labels,
            return_dict=True,
        )
        return out.loss

    @torch.no_grad()
    def generate(self, input_features, **kwargs):
        h = self.encode_with_adapter(input_features)
        return self.whisper.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=h),
            **kwargs,
        )


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum_steps, use_amp):
    model.train()
    total_loss = 0.0
    step_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train")):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(input_features=input_features, labels=labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        step_count += 1

    return total_loss / max(step_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    step_count = 0
    for batch in tqdm(loader, desc="valid_loss"):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(input_features=input_features, labels=labels)
        total_loss += loss.item()
        step_count += 1
    return total_loss / max(step_count, 1)


@torch.no_grad()
def evaluate_validation_wer(model, loader, processor, device, max_new_tokens):
    model.eval()
    refs = []
    preds = []
    for batch in tqdm(loader, desc="valid_wer"):
        input_features = batch["input_features"].to(device)
        pred_ids = model.generate(
            input_features=input_features,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        batch_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        preds.extend(normalize_text(p) for p in batch_preds)
        refs.extend(normalize_text(t) for t in batch["texts"])
    if not refs:
        return float("inf")
    return jiwer_wer(refs, preds)


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--adapter_bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_scale_init", type=float, default=0.01)
    parser.add_argument("--adapter_style", type=str, default="kaust_layerwise", choices=["kaust_layerwise", "single_encoder_output"])
    parser.add_argument("--train_whisper", action="store_true", help="If set, also update Whisper weights. Default freezes Whisper.")
    parser.add_argument("--selection_max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("dataset_type:", args.dataset_type)
    print("model selection: validation WER")

    processor = WhisperProcessor.from_pretrained(args.whisper_name, language="en", task="transcribe")
    train_records = load_records_from_args(args, "train", save_dir=args.save_dir)
    valid_records = load_records_from_args(args, "valid", save_dir=args.save_dir)

    train_loader = DataLoader(
        ASRDataset(train_records, processor, audio_ext=args.audio_ext),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ASRCollator(processor),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        ASRDataset(valid_records, processor, audio_ext=args.audio_ext),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ASRCollator(processor),
        pin_memory=True,
    )

    model = ResidualAdapterWhisper(
        whisper_name=args.whisper_name,
        adapter_bottleneck=args.adapter_bottleneck,
        dropout=args.dropout,
        adapter_scale_init=args.adapter_scale_init,
        freeze_whisper=not args.train_whisper,
        adapter_style=args.adapter_style,
    ).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("adapter_style:", args.adapter_style)
    print("num_encoder_layers:", len(model.whisper.model.encoder.layers))
    print("total params:", sum(p.numel() for p in model.parameters()))
    print("trainable params:", sum(p.numel() for p in trainable_params))
    print("freeze_whisper:", not args.train_whisper)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs // max(args.grad_accum_steps, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_valid_wer = float("inf")
    rows = []
    log_path = os.path.join(args.save_dir, "train_log.csv")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.grad_accum_steps, use_amp)
        valid_loss = evaluate_loss(model, valid_loader, device, use_amp)
        valid_wer = evaluate_validation_wer(model, valid_loader, processor, device, args.selection_max_new_tokens)

        print(f"train_loss: {train_loss:.6f}")
        print(f"valid_loss: {valid_loss:.6f}")
        print(f"valid_wer: {valid_wer:.6f}")
        rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_wer": valid_wer})
        pd.DataFrame(rows).to_csv(log_path, index=False)

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "adapter_state_dict": model.adapter_state_dict(),
            "adapter_style": args.adapter_style,
            "num_encoder_layers": len(model.whisper.model.encoder.layers),
            "valid_loss": valid_loss,
            "valid_wer": valid_wer,
            "selection_metric": "valid_wer",
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if valid_wer < best_valid_wer:
            best_valid_wer = valid_wer
            ckpt["best_valid_wer"] = best_valid_wer
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print("saved best by valid_wer:", os.path.join(args.save_dir, "best.pt"))

    print("best_valid_wer:", best_valid_wer)


if __name__ == "__main__":
    main()
