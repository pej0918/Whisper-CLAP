import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
from jiwer import cer, wer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
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
        required = ["sample_id", "video_id", "audio_path", "start", "end", "duration", "text_spoken"]
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
            "sample_id": str(r["sample_id"]),
            "video_id": str(r["video_id"]),
            "audio": load_segment(r["audio_path"], r["start"], r["end"]),
            "start": float(r["start"]),
            "end": float(r["end"]),
            "duration": float(r["duration"]),
            "reference": str(r["text_spoken"]),
        }


def collate_fn(batch):
    return batch


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
    def __init__(self, whisper_name="openai/whisper-base", adapter_bottleneck=256, dropout=0.0):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        hidden_dim = self.whisper.config.d_model
        encoder_layers = self.whisper.model.encoder.layers
        self.adapters = nn.ModuleList([
            KAUSTStyleResidualAdapter(hidden_dim, adapter_bottleneck, dropout)
            for _ in range(len(encoder_layers))
        ])
        self._adapter_hooks = []
        for layer_idx, layer in enumerate(encoder_layers):
            self._adapter_hooks.append(layer.register_forward_hook(self._make_layerwise_adapter_hook(layer_idx)))
        for p in self.whisper.parameters():
            p.requires_grad = False

    def _make_layerwise_adapter_hook(self, layer_idx):
        def hook(module, inputs, output):
            adapted = self.adapters[layer_idx](output[0]) if isinstance(output, tuple) else self.adapters[layer_idx](output)
            if isinstance(output, tuple):
                return (adapted,) + output[1:]
            return adapted
        return hook

    def load_adapter_state_dict(self, state_dict, strict=True):
        return self.adapters.load_state_dict(state_dict, strict=strict)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--model", default="openai/whisper-base")
    parser.add_argument("--adapter_bottleneck", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model, language="English", task="transcribe")
    model = ResidualAdapterWhisper(args.model, args.adapter_bottleneck).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_adapter_state_dict(ckpt["adapter_state_dict"], strict=True)
    model.whisper.to(dtype=dtype)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    dataset = ManifestDataset(args.manifest, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    rows = []
    total_audio_sec = 0.0
    tic = time.time()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Residual inference"):
            audios = [x["audio"] for x in batch]
            inputs = processor.feature_extractor(audios, sampling_rate=TARGET_SR, return_tensors="pt")
            input_features = inputs.input_features.to(device=device, dtype=dtype)
            pred_ids = model.generate(
                input_features=input_features,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
            for item, pred in zip(batch, predictions):
                ref_raw = item["reference"].strip()
                pred_raw = pred.strip()
                ref_norm = normalize_text(processor.tokenizer, ref_raw)
                pred_norm = normalize_text(processor.tokenizer, pred_raw)
                sample_wer = wer(ref_norm, pred_norm) if ref_norm else np.nan
                sample_cer = cer(ref_norm, pred_norm) if ref_norm else np.nan
                rows.append({
                    "sample_id": item["sample_id"],
                    "video_id": item["video_id"],
                    "start": item["start"],
                    "end": item["end"],
                    "duration": item["duration"],
                    "reference": ref_raw,
                    "prediction": pred_raw,
                    "reference_norm": ref_norm,
                    "prediction_norm": pred_norm,
                    "sample_wer": sample_wer,
                    "sample_cer": sample_cer,
                })
                total_audio_sec += item["duration"]

    elapsed = time.time() - tic
    out = pd.DataFrame(rows)
    valid = out[out["reference_norm"].fillna("").str.len() > 0].copy()
    refs = valid["reference_norm"].tolist()
    hyps = valid["prediction_norm"].tolist()
    corpus_wer = wer(refs, hyps)
    corpus_cer = cer(refs, hyps)
    summary = {
        "model": args.model,
        "method": "KAUST-style Residual Adapter",
        "checkpoint": args.ckpt,
        "manifest": args.manifest,
        "samples": int(len(out)),
        "valid_samples": int(len(valid)),
        "audio_hours": float(total_audio_sec / 3600.0),
        "wer": float(corpus_wer),
        "cer": float(corpus_cer),
        "num_beams": int(args.num_beams),
        "batch_size": int(args.batch_size),
        "elapsed_seconds": float(elapsed),
        "rtf": float(elapsed / max(total_audio_sec, 1e-8)),
        "best_dev_wer": ckpt.get("best_valid_wer", ckpt.get("valid_wer")),
        "adapter_bottleneck": args.adapter_bottleneck,
    }

    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("FINAL RESULT")
    print(f"Samples : {len(out)}")
    print(f"WER     : {corpus_wer:.4f}")
    print(f"CER     : {corpus_cer:.4f}")
    print(f"RTF     : {summary['rtf']:.4f}")
    print("CSV     :", output_csv)
    print("Summary :", summary_json)


if __name__ == "__main__":
    main()
