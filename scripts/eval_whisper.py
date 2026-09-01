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
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

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


def load_segment(path, start, end):
    start = float(start)
    end = float(end)

    if end <= start:
        raise ValueError(
            f"Invalid segment: start={start}, end={end}, path={path}"
        )

    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate

        start_frame = int(round(start * sr))
        n_frames = int(round((end - start) * sr))

        f.seek(start_frame)

        wav = f.read(
            frames=n_frames,
            dtype="float32",
            always_2d=True,
        )

    if wav.shape[0] == 0:
        raise RuntimeError(
            f"Empty segment: {path} [{start:.3f}, {end:.3f}]"
        )

    # [time, channel] -> mono
    wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        wav = torch.from_numpy(wav)
        wav = AF.resample(wav, sr, TARGET_SR)
        wav = wav.numpy()

    return wav.astype(np.float32)


class M3AVDataset(Dataset):
    def __init__(self, manifest, max_samples=None):
        self.df = pd.read_csv(manifest)

        required = [
            "sample_id",
            "video_id",
            "audio_path",
            "start",
            "end",
            "duration",
            "text_spoken",
        ]

        missing = [x for x in required if x not in self.df.columns]

        if missing:
            raise ValueError(
                f"Manifest missing columns: {missing}"
            )

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].copy()

        self.df = self.df.reset_index(drop=True)

        print("Manifest :", manifest)
        print("Samples  :", len(self.df))
        print(
            "Hours    :",
            round(self.df["duration"].sum() / 3600, 3)
        )
        print(
            "Max dur  :",
            round(self.df["duration"].max(), 3),
            "sec"
        )

        n_long = int((self.df["duration"] > 30.0).sum())
        print(">30 sec  :", n_long)

        if n_long > 0:
            raise RuntimeError(
                f"{n_long} segments exceed 30 seconds. "
                "Do not silently truncate them."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        wav = load_segment(
            r["audio_path"],
            r["start"],
            r["end"],
        )

        return {
            "sample_id": str(r["sample_id"]),
            "video_id": str(r["video_id"]),
            "audio": wav,
            "start": float(r["start"]),
            "end": float(r["end"]),
            "duration": float(r["duration"]),
            "reference": str(r["text_spoken"]),
        }


def collate_fn(batch):
    return batch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)

    parser.add_argument(
        "--model",
        default="openai/whisper-base"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=5
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None
    )

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print("=" * 70)
    print("M3AV WHISPER EVALUATION")
    print("=" * 70)

    print("model      :", args.model)
    print("device     :", device)
    print("dtype      :", dtype)
    print("batch size :", args.batch_size)
    print("num beams  :", args.num_beams)

    processor = WhisperProcessor.from_pretrained(
        args.model
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english",
        task="transcribe",
    )

    dataset = M3AVDataset(
        args.manifest,
        max_samples=args.max_samples,
    )

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

        for batch in tqdm(
            loader,
            desc="Whisper inference"
        ):
            audios = [
                x["audio"]
                for x in batch
            ]

            inputs = processor.feature_extractor(
                audios,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )

            input_features = (
                inputs.input_features
                .to(device=device, dtype=dtype)
            )

            pred_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )

            predictions = processor.batch_decode(
                pred_ids,
                skip_special_tokens=True,
            )

            for item, pred in zip(batch, predictions):

                ref_raw = item["reference"].strip()
                pred_raw = pred.strip()

                ref_norm = normalize_text(
                    processor.tokenizer,
                    ref_raw,
                )

                pred_norm = normalize_text(
                    processor.tokenizer,
                    pred_raw,
                )

                if ref_norm:
                    sample_wer = wer(
                        ref_norm,
                        pred_norm,
                    )

                    sample_cer = cer(
                        ref_norm,
                        pred_norm,
                    )
                else:
                    sample_wer = np.nan
                    sample_cer = np.nan

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

    valid = out[
        out["reference_norm"].fillna("").str.len() > 0
    ].copy()

    refs = valid["reference_norm"].tolist()
    hyps = valid["prediction_norm"].tolist()

    corpus_wer = wer(refs, hyps)
    corpus_cer = cer(refs, hyps)

    summary = {
        "model": args.model,
        "manifest": args.manifest,

        "samples": int(len(out)),
        "valid_samples": int(len(valid)),

        "audio_hours": float(
            total_audio_sec / 3600.0
        ),

        "wer": float(corpus_wer),
        "cer": float(corpus_cer),

        "num_beams": int(args.num_beams),
        "batch_size": int(args.batch_size),

        "elapsed_seconds": float(elapsed),
        "rtf": float(
            elapsed / max(total_audio_sec, 1e-8)
        ),
    }

    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_csv,
        index=False,
    )

    with open(
        summary_json,
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Samples : {len(out)}"
    )

    print(
        f"Hours   : {total_audio_sec / 3600:.3f}"
    )

    print(
        f"WER     : {corpus_wer:.4f}"
    )

    print(
        f"CER     : {corpus_cer:.4f}"
    )

    print(
        f"RTF     : {summary['rtf']:.4f}"
    )

    print(
        f"Elapsed : {elapsed / 60:.1f} min"
    )

    print(
        "CSV     :",
        output_csv
    )

    print(
        "Summary :",
        summary_json
    )


if __name__ == "__main__":
    main()
