import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
from jiwer import wer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup, set_seed
from transformers.modeling_outputs import BaseModelOutput

TARGET_SR = 16000


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
        wav = AF.resample(wav, sr, TARGET_SR).numpy()
    return wav.astype(np.float32)


class ManifestDataset(Dataset):
    def __init__(self, manifest, max_samples=None):
        self.df = pd.read_csv(manifest)
        required = ["audio_path", "start", "end", "duration", "text_spoken"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].copy()
        self.df = self.df.reset_index(drop=True)
        n_long = int((self.df["duration"] > 30.0).sum())
        if n_long > 0:
            raise RuntimeError(f"{n_long} segments > 30 sec")
        print(f"{manifest}\n  samples = {len(self.df)}\n  hours   = {self.df['duration'].sum()/3600:.3f}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return {
            "audio": load_segment(r["audio_path"], r["start"], r["end"]),
            "text": str(r["text_spoken"]),
        }


class WhisperCollator:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        audios = [x["audio"] for x in features]
        texts = [x["text"] for x in features]
        audio_batch = self.processor.feature_extractor(audios, sampling_rate=TARGET_SR, return_tensors="pt")
        label_batch = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = label_batch["input_ids"].masked_fill(label_batch["attention_mask"].ne(1), -100)
        if (
            labels.ndim == 2
            and labels.shape[1] > 0
            and self.decoder_start_token_id is not None
            and (labels[:, 0] == self.decoder_start_token_id).all().item()
        ):
            labels = labels[:, 1:]
        return {"input_features": audio_batch["input_features"], "labels": labels, "texts": texts}


class KAUSTStyleResidualAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=256, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, h):
        return h + self.up(self.dropout(self.act(self.down(h))))


class ResidualAdapterWhisper(nn.Module):
    def __init__(self, whisper_name="openai/whisper-base", adapter_bottleneck=256, dropout=0.0, freeze_whisper=True):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        self.whisper.config.use_cache = False
        hidden_dim = self.whisper.config.d_model
        encoder_layers = self.whisper.model.encoder.layers
        self.adapters = nn.ModuleList([
            KAUSTStyleResidualAdapter(hidden_dim, adapter_bottleneck, dropout)
            for _ in range(len(encoder_layers))
        ])
        self._adapter_hooks = []
        for layer_idx, layer in enumerate(encoder_layers):
            self._adapter_hooks.append(layer.register_forward_hook(self._make_layerwise_adapter_hook(layer_idx)))
        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad = False
        for p in self.adapters.parameters():
            p.requires_grad = True

    def _make_layerwise_adapter_hook(self, layer_idx):
        def hook(module, inputs, output):
            adapted = self.adapters[layer_idx](output[0]) if isinstance(output, tuple) else self.adapters[layer_idx](output)
            if isinstance(output, tuple):
                return (adapted,) + output[1:]
            return adapted
        return hook

    def adapter_state_dict(self):
        return self.adapters.state_dict()

    def load_adapter_state_dict(self, state_dict, strict=True):
        return self.adapters.load_state_dict(state_dict, strict=strict)

    def forward(self, input_features, labels):
        enc = self.whisper.model.encoder(input_features, return_dict=True)
        return self.whisper(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
            labels=labels,
            return_dict=True,
        ).loss

    @torch.no_grad()
    def generate(self, input_features, **kwargs):
        enc = self.whisper.model.encoder(input_features, return_dict=True)
        return self.whisper.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
            **kwargs,
        )


def normalize_text(tokenizer, text):
    text = str(text).strip()
    if hasattr(tokenizer, "normalize"):
        return tokenizer.normalize(text).strip()
    if hasattr(tokenizer, "_normalize"):
        return tokenizer._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found.")


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum_steps, use_amp):
    model.train()
    total_loss = 0.0
    steps = 0
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
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(loader, desc="valid_loss"):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(input_features=input_features, labels=labels)
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_validation_wer(model, loader, processor, device, num_beams, max_new_tokens):
    model.eval()
    refs, preds = [], []
    for batch in tqdm(loader, desc="valid_wer"):
        input_features = batch["input_features"].to(device)
        pred_ids = model.generate(
            input_features=input_features,
            num_beams=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        batch_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        preds.extend(normalize_text(processor.tokenizer, p) for p in batch_preds)
        refs.extend(normalize_text(processor.tokenizer, t) for t in batch["texts"])
    filtered = [(r, p) for r, p in zip(refs, preds) if r]
    if not filtered:
        return float("inf")
    return wer([x[0] for x in filtered], [x[1] for x in filtered])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model", default="openai/whisper-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--adapter_bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--selection_num_beams", type=int, default=5)
    parser.add_argument("--selection_max_new_tokens", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained(args.model, language="English", task="transcribe")

    train_ds = ManifestDataset(args.train, args.max_train_samples)
    dev_ds = ManifestDataset(args.dev, args.max_dev_samples)
    collator = WhisperCollator(processor, WhisperForConditionalGeneration.from_pretrained(args.model).config.decoder_start_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator, pin_memory=True)

    model = ResidualAdapterWhisper(args.model, args.adapter_bottleneck, args.dropout, freeze_whisper=True).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print("=" * 70)
    print("KAUST-STYLE RESIDUAL ADAPTER")
    print("model              :", args.model)
    print("freeze_whisper     : True")
    print("num_encoder_layers :", len(model.whisper.model.encoder.layers))
    print("bottleneck         :", args.adapter_bottleneck)
    print("learning rate      :", args.lr)
    print("epochs             :", args.epochs)
    print("trainable params   :", f"{trainable_count:,}", f"({trainable_count/1e6:.3f}M)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs // max(args.grad_accum_steps, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_valid_wer = float("inf")
    rows = []
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.grad_accum_steps, use_amp)
        valid_loss = evaluate_loss(model, dev_loader, device, use_amp)
        valid_wer = evaluate_validation_wer(model, dev_loader, processor, device, args.selection_num_beams, args.selection_max_new_tokens)
        print(f"train_loss: {train_loss:.6f}")
        print(f"valid_loss: {valid_loss:.6f}")
        print(f"valid_wer : {valid_wer:.6f}")
        rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_wer": valid_wer})
        pd.DataFrame(rows).to_csv(save_dir / "train_log.csv", index=False)
        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "adapter_state_dict": model.adapter_state_dict(),
            "num_encoder_layers": len(model.whisper.model.encoder.layers),
            "adapter_bottleneck": args.adapter_bottleneck,
            "trainable_params": trainable_count,
            "valid_loss": valid_loss,
            "valid_wer": valid_wer,
            "selection_metric": "valid_wer",
        }
        torch.save(ckpt, save_dir / "last.pt")
        if valid_wer < best_valid_wer:
            best_valid_wer = valid_wer
            ckpt["best_valid_wer"] = best_valid_wer
            torch.save(ckpt, save_dir / "best.pt")
            print("saved best:", save_dir / "best.pt")

    summary = {
        "base_model": args.model,
        "method": "KAUST-style Residual Adapter",
        "freeze_whisper": True,
        "adapter_bottleneck": args.adapter_bottleneck,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "trainable_params": trainable_count,
        "trainable_params_M": trainable_count / 1e6,
        "best_dev_wer": best_valid_wer,
        "selection_num_beams": args.selection_num_beams,
        "selection_max_new_tokens": args.selection_max_new_tokens,
        "seed": args.seed,
    }
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("best_valid_wer:", best_valid_wer)


if __name__ == "__main__":
    main()
