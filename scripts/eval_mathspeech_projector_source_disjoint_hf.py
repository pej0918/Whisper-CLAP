import argparse
import json
import time
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

from train_mathspeech_projector_source_disjoint_hf import WhisperSemanticASR, torch_load_compat

TARGET_SR = 16000


def load_audio(path):
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        wav = f.read(dtype="float32", always_2d=True)
    if wav.shape[0] == 0:
        raise RuntimeError(f"Empty audio: {path}")
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = AF.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy()
    wav = wav.astype(np.float32)
    duration = len(wav) / TARGET_SR
    if duration > 30.0:
        raise RuntimeError(f"Audio exceeds 30 sec ({duration:.3f}s): {path}")
    return wav, duration


def normalize_text(processor, text):
    tok = processor.tokenizer
    text = str(text).strip()
    if hasattr(tok, "normalize"):
        return tok.normalize(text).strip()
    if hasattr(tok, "_normalize"):
        return tok._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found")


class EvalDataset(Dataset):
    def __init__(self, manifest, processor):
        self.df = pd.read_csv(manifest).reset_index(drop=True)
        self.processor = processor
        required = ["sample_id", "audio_path", "reference_text"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav, duration = load_audio(str(r["audio_path"]))
        feat = self.processor.feature_extractor(
            wav, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features[0]
        return {
            "input_features": feat,
            "sample_id": str(r["sample_id"]),
            "source": str(r["source"]) if "source" in self.df.columns else "",
            "duration": duration,
            "reference": str(r["reference_text"]),
        }


def collate(batch):
    return {
        "input_features": torch.stack([x["input_features"] for x in batch]),
        "sample_id": [x["sample_id"] for x in batch],
        "source": [x["source"] for x in batch],
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
    freeze_whisper = bool(cargs.get("freeze_whisper", True))
    alignment_mode = cargs.get("alignment_mode", "absolute")

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
        freeze_whisper=freeze_whisper,
    )

    # Full checkpoints are authoritative. This is required for CLAP-guided
    # full fine-tuning, where Whisper itself changes during training.
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        state_source = "full_model_state_dict"
    elif "adapter_state_dict" in ckpt:
        # Backward compatibility for compact frozen-Whisper checkpoints.
        model.adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
        model.pooler.load_state_dict(ckpt["pooler_state_dict"], strict=True)
        model.align_head.load_state_dict(ckpt["align_head_state_dict"], strict=True)
        state_source = "compact_adapter_states"
    else:
        raise ValueError("Checkpoint contains neither model_state_dict nor compact adapter states")

    print("checkpoint freeze_whisper:", freeze_whisper)
    print("checkpoint alignment_mode:", alignment_mode)
    print("loaded state source      :", state_source)

    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_ids
    model.whisper.generation_config.forced_decoder_ids = forced_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")
    model = model.to(device)
    model.eval()

    ds = EvalDataset(args.manifest, processor)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
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
                    "source": batch["source"][i],
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
        "wer": float(corpus_wer),
        "cer": float(corpus_cer),
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "rtf": rtf,
        "adapter_type": cargs.get("adapter_type", "gated"),
        "pool_type": cargs.get("pool_type", "mean"),
        "alignment_mode": alignment_mode,
        "lambda_align": cargs.get("lambda_align"),
        "lambda_hidden": cargs.get("lambda_hidden"),
        "align_loss_type": cargs.get("align_loss_type"),
        "freeze_whisper": freeze_whisper,
        "state_source": state_source,
        "cleaning": False,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("predictions:", out_csv)
    print("summary    :", out_json)


if __name__ == "__main__":
    main()
