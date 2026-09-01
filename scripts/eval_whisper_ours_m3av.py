import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from jiwer import wer, cer

from transformers import WhisperProcessor

# Import the exact model implementation used for training.
from train_whisper_ours_m3av import WhisperSemanticASR, torch_load_compat

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
        wav = AF.resample(wav, sr, TARGET_SR)
        wav = wav.numpy()
    return wav.astype(np.float32)


def normalize_text(processor, text):
    tok = processor.tokenizer
    text = str(text).strip()
    if hasattr(tok, "normalize"):
        return tok.normalize(text).strip()
    if hasattr(tok, "_normalize"):
        return tok._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found.")


class EvalDataset(Dataset):
    def __init__(self, manifest, processor):
        self.df = pd.read_csv(manifest).reset_index(drop=True)
        self.processor = processor
        required = ["sample_id", "video_id", "audio_path", "start", "end", "duration", "text_spoken"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing manifest columns: {missing}")
        if (self.df["duration"].astype(float) > 30.0).any():
            raise RuntimeError("Found segment >30 sec.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav = load_segment(r["audio_path"], r["start"], r["end"])
        feat = self.processor.feature_extractor(
            wav, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features[0]
        return {
            "input_features": feat,
            "sample_id": str(r["sample_id"]),
            "video_id": str(r["video_id"]),
            "start": float(r["start"]),
            "end": float(r["end"]),
            "duration": float(r["duration"]),
            "reference": str(r["text_spoken"]),
        }


def collate(batch):
    return {
        "input_features": torch.stack([x["input_features"] for x in batch]),
        "sample_id": [x["sample_id"] for x in batch],
        "video_id": [x["video_id"] for x in batch],
        "start": [x["start"] for x in batch],
        "end": [x["end"] for x in batch],
        "duration": [x["duration"] for x in batch],
        "reference": [x["reference"] for x in batch],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    ckpt = torch_load_compat(args.ckpt, map_location="cpu")
    cargs = ckpt.get("args", {})
    clap_dim = int(ckpt.get("clap_dim", 512))
    whisper_name = cargs.get("whisper_name", "openai/whisper-base")

    processor = WhisperProcessor.from_pretrained(
        whisper_name, language="English", task="transcribe"
    )
    model = WhisperSemanticASR(
        whisper_name=whisper_name,
        clap_dim=clap_dim,
        adapter_type=cargs.get("adapter_type", "gated"),
        pool_type=cargs.get("pool_type", "mean"),
        adapter_bottleneck=int(cargs.get("adapter_bottleneck", 256)),
        dropout=float(cargs.get("dropout", 0.1)),
        adapter_scale_init=float(cargs.get("adapter_scale_init", 0.01)),
    )
    model.adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
    model.pooler.load_state_dict(ckpt["pooler_state_dict"], strict=True)
    model.align_head.load_state_dict(ckpt["align_head_state_dict"], strict=True)

    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_ids
    model.whisper.generation_config.forced_decoder_ids = forced_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")
    model = model.to(device)
    model.eval()

    ds = EvalDataset(args.manifest, processor)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=True,
    )

    rows = []
    refs_norm, hyps_norm = [], []
    total_audio_sec = 0.0
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            feats = batch["input_features"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_ids = model.generate(
                    input_features=feats,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for i, pred in enumerate(pred_text):
                ref = batch["reference"][i]
                rnorm = normalize_text(processor, ref)
                pnorm = normalize_text(processor, pred)
                if not rnorm:
                    continue
                refs_norm.append(rnorm)
                hyps_norm.append(pnorm)
                dur = float(batch["duration"][i])
                total_audio_sec += dur
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "video_id": batch["video_id"][i],
                    "start": batch["start"][i],
                    "end": batch["end"][i],
                    "duration": dur,
                    "reference": ref,
                    "prediction": pred,
                    "reference_norm": rnorm,
                    "prediction_norm": pnorm,
                    "sample_wer": wer(rnorm, pnorm),
                    "sample_cer": cer(rnorm, pnorm),
                })

    elapsed = time.time() - t0
    corpus_wer = wer(refs_norm, hyps_norm)
    corpus_cer = cer(refs_norm, hyps_norm)
    rtf = elapsed / total_audio_sec if total_audio_sec > 0 else None

    out_csv = Path(args.output_csv)
    out_json = Path(args.summary_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        "model": "WhisperSemanticASR",
        "base_model": whisper_name,
        "checkpoint": args.ckpt,
        "manifest": args.manifest,
        "samples": len(ds),
        "valid_samples": len(rows),
        "audio_hours": total_audio_sec / 3600.0,
        "wer": corpus_wer,
        "cer": corpus_cer,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "rtf": rtf,
        "adapter_type": cargs.get("adapter_type", "gated"),
        "pool_type": cargs.get("pool_type", "mean"),
        "lambda_align": cargs.get("lambda_align"),
        "lambda_hidden": cargs.get("lambda_hidden"),
        "align_loss_type": cargs.get("align_loss_type"),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("predictions:", out_csv)
    print("summary    :", out_json)


if __name__ == "__main__":
    main()
