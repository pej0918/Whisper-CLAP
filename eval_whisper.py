import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from jiwer import cer, wer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr_dataset_utils import add_dataset_args, load_record_audio_16k, load_records_from_args

TARGET_SR = 16000


def normalize_text(tokenizer, text):
    text = str(text).strip()
    if hasattr(tokenizer, "normalize"):
        return tokenizer.normalize(text).strip()
    if hasattr(tokenizer, "_normalize"):
        return tokenizer._normalize(text).strip()
    raise RuntimeError(
        "Whisper tokenizer normalizer not found. "
        "Do not silently use a different normalization."
    )


class EvalASRDataset(Dataset):
    def __init__(self, records, max_samples: Optional[int] = None):
        self.records = list(records)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        print("Samples  :", len(self.records))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        wav = load_record_audio_16k(rec)
        duration = float(len(wav) / TARGET_SR)
        return {
            "uid": str(rec.uid),
            "audio": wav,
            "duration": duration,
            "reference": str(rec.text),
            "metadata": rec.metadata or {},
        }


def collate_fn(batch):
    return batch


def get_forced_decoder_ids(processor):
    # Do NOT also write this into model.config/generation_config.
    # Passing forced_decoder_ids to generate is enough. Writing both places
    # creates a duplicated ForceTokensLogitsProcessor on some transformers versions.
    return processor.get_decoder_prompt_ids(
        language="english",
        task="transcribe",
    )


def load_hf_model(args, dtype):
    model = WhisperForConditionalGeneration.from_pretrained(
        args.whisper_name,
        torch_dtype=dtype,
    )
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        print("loaded checkpoint:", args.ckpt_path)
        print("selection_metric:", ckpt.get("selection_metric"))
        print("valid_wer:", ckpt.get("valid_wer"))
    return model


def load_lora_model(args, dtype):
    from peft import PeftModel

    if not args.adapter_dir:
        raise ValueError("--adapter_dir is required for --model_kind lora")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.whisper_name,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    print("loaded LoRA adapter:", args.adapter_dir)
    return model


def load_residual_adapter_model(args, dtype):
    from train_whisper_residual_adapter import ResidualAdapterWhisper

    if not args.ckpt_path:
        raise ValueError("--ckpt_path is required for --model_kind residual_adapter")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    adapter_style = ckpt.get("adapter_style", train_args.get("adapter_style", "single_encoder_output"))
    model = ResidualAdapterWhisper(
        whisper_name=train_args.get("whisper_name", args.whisper_name),
        adapter_bottleneck=int(train_args.get("adapter_bottleneck", args.adapter_bottleneck)),
        dropout=float(train_args.get("dropout", args.dropout)),
        adapter_scale_init=float(train_args.get("adapter_scale_init", args.adapter_scale_init)),
        freeze_whisper=True,
        adapter_style=adapter_style,
    )
    model.load_adapter_state_dict(ckpt["adapter_state_dict"], strict=True)
    model.to(dtype=dtype)
    print("loaded residual adapter checkpoint:", args.ckpt_path)
    print("adapter_style:", adapter_style)
    print("selection_metric:", ckpt.get("selection_metric"))
    print("valid_wer:", ckpt.get("valid_wer"))
    return model


def load_clap_adapter_model(args, dtype):
    from train_whisper_clap_adapter import WhisperSemanticASR

    if not args.ckpt_path:
        raise ValueError("--ckpt_path is required for --model_kind clap_adapter")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    whisper_name = train_args.get("whisper_name", args.whisper_name)
    clap_dim = int(ckpt.get("clap_dim", 512))
    model = WhisperSemanticASR(
        whisper_name=whisper_name,
        clap_dim=clap_dim,
        adapter_type=train_args.get("adapter_type", "gated"),
        pool_type=train_args.get("pool_type", "mean"),
        adapter_bottleneck=int(train_args.get("adapter_bottleneck", 256)),
        dropout=float(train_args.get("dropout", 0.1)),
        adapter_scale_init=float(train_args.get("adapter_scale_init", 0.01)),
        freeze_whisper=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(dtype=dtype)
    print("loaded CLAP adapter checkpoint:", args.ckpt_path)
    print("selection_metric:", ckpt.get("selection_metric"))
    print("valid_wer:", ckpt.get("valid_wer"))
    return model


def load_model(args, dtype):
    if args.model_kind == "hf":
        return load_hf_model(args, dtype)
    if args.model_kind == "lora":
        return load_lora_model(args, dtype)
    if args.model_kind == "residual_adapter":
        return load_residual_adapter_model(args, dtype)
    if args.model_kind == "clap_adapter":
        return load_clap_adapter_model(args, dtype)
    raise ValueError(f"Unknown model_kind: {args.model_kind}")


@torch.no_grad()
def run_eval(args, model, processor, loader, device, dtype, forced_decoder_ids):
    rows: List[Dict] = []
    total_audio_sec = 0.0
    tic = time.time()

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Whisper inference"):
            audios = [x["audio"] for x in batch]
            inputs = processor.feature_extractor(
                audios,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device=device, dtype=dtype)

            pred_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=args.num_beams,
                do_sample=False,
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

                if ref_norm:
                    sample_wer = wer(ref_norm, pred_norm)
                    sample_cer = cer(ref_norm, pred_norm)
                else:
                    sample_wer = np.nan
                    sample_cer = np.nan

                row = {
                    "uid": item["uid"],
                    "duration": item["duration"],
                    "transcription": ref_raw,
                    "reference": ref_raw,
                    "prediction": pred_raw,
                    args.pred_col: pred_raw,
                    "reference_norm": ref_norm,
                    "prediction_norm": pred_norm,
                    "sample_wer": sample_wer,
                    "sample_cer": sample_cer,
                }
                meta = item.get("metadata") or {}
                if isinstance(meta, dict):
                    row.update({f"meta_{k}": v for k, v in meta.items()})
                rows.append(row)
                total_audio_sec += float(item["duration"])

    elapsed = time.time() - tic
    out = pd.DataFrame(rows)
    valid = out[out["reference_norm"].fillna("").str.len() > 0].copy()
    refs = valid["reference_norm"].tolist()
    hyps = valid["prediction_norm"].tolist()

    corpus_wer = wer(refs, hyps) if refs else float("nan")
    corpus_cer = cer(refs, hyps) if refs else float("nan")
    summary = {
        "model_kind": args.model_kind,
        "model": args.whisper_name,
        "checkpoint": args.ckpt_path,
        "adapter_dir": args.adapter_dir,
        "dataset_type": args.dataset_type,
        "eval_split": args.eval_split,
        "samples": int(len(out)),
        "valid_samples": int(len(valid)),
        "audio_hours": float(total_audio_sec / 3600.0),
        "wer": float(corpus_wer),
        "cer": float(corpus_cer),
        "num_beams": int(args.num_beams),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "elapsed_seconds": float(elapsed),
        "rtf": float(elapsed / max(total_audio_sec, 1e-8)),
        "output_csv": str(args.output_csv),
    }
    return out, summary


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument("--model_kind", type=str, default="hf", choices=["hf", "lora", "residual_adapter", "clap_adapter"])
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--pred_col", type=str, default="prediction")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter_bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_scale_init", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (device.type == "cuda" or args.fp16) else torch.float32

    print("=" * 70)
    print("WHISPER EVALUATION")
    print("=" * 70)
    print("model_kind :", args.model_kind)
    print("model      :", args.whisper_name)
    print("dataset    :", args.dataset_type)
    print("split      :", args.eval_split)
    print("device     :", device)
    print("dtype      :", dtype)
    print("batch size :", args.batch_size)
    print("num beams  :", args.num_beams)

    processor = WhisperProcessor.from_pretrained(args.whisper_name)
    records = load_records_from_args(args, args.eval_split, save_dir=os.path.dirname(args.output_csv) or ".")
    dataset = EvalASRDataset(records, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = load_model(args, dtype=dtype)
    model = model.to(device)
    forced_decoder_ids = get_forced_decoder_ids(processor)

    out, summary = run_eval(args, model, processor, loader, device, dtype, forced_decoder_ids)

    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Samples : {summary['samples']}")
    print(f"Hours   : {summary['audio_hours']:.3f}")
    print(f"WER     : {summary['wer']:.4f}")
    print(f"CER     : {summary['cer']:.4f}")
    print(f"RTF     : {summary['rtf']:.4f}")
    print(f"Elapsed : {summary['elapsed_seconds'] / 60:.1f} min")
    print("CSV     :", output_csv)
    print("Summary :", summary_json)


if __name__ == "__main__":
    main()
