import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

from torch.utils.data import Dataset
from jiwer import wer, cer

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

TARGET_SR = 16000


def load_segment(path, start, end):
    start = float(start)
    end = float(end)

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
            f"Empty audio: {path} [{start}, {end}]"
        )

    wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        wav = torch.from_numpy(wav)
        wav = AF.resample(wav, sr, TARGET_SR)
        wav = wav.numpy()

    return wav.astype(np.float32)


class M3AVTrainDataset(Dataset):
    def __init__(self, manifest, max_samples=None):
        self.df = pd.read_csv(manifest)

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].copy()

        self.df = self.df.reset_index(drop=True)

        if (self.df["duration"] > 30.0).any():
            raise RuntimeError(
                "Found segment >30 sec."
            )

        print(
            f"{manifest}: "
            f"{len(self.df)} samples, "
            f"{self.df.duration.sum()/3600:.2f} h"
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
            "audio": wav,
            "text": str(r["text_spoken"]),
        }


class WhisperCollator:
    def __init__(
        self,
        processor,
        decoder_start_token_id,
    ):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):

        audios = [
            x["audio"]
            for x in features
        ]

        texts = [
            x["text"]
            for x in features
        ]

        batch = self.processor.feature_extractor(
            audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        # Whisper adds decoder_start_token internally.
        if (
            labels.shape[1] > 0
            and
            (labels[:, 0] == self.decoder_start_token_id)
            .all()
            .item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument(
        "--model",
        default="openai/whisper-base",
    )

    ap.add_argument(
        "--epochs",
        type=float,
        default=3,
    )

    ap.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
    )

    ap.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--max_dev_samples",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(
        args.model,
        language="English",
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 70)
    print("FULL FINE-TUNING")
    print("=" * 70)
    print(f"trainable params: {trainable:,} ({trainable/1e6:.3f}M)")
    print(f"all params      : {total:,}")
    print(f"trainable %     : {100*trainable/total:.3f}%")

    forced_ids = processor.get_decoder_prompt_ids(
        language="english",
        task="transcribe",
    )

    model.generation_config.forced_decoder_ids = forced_ids

    train_ds = M3AVTrainDataset(
        args.train,
        args.max_train_samples,
    )

    dev_ds = M3AVTrainDataset(
        args.dev,
        args.max_dev_samples,
    )

    collator = WhisperCollator(
        processor,
        model.config.decoder_start_token_id,
    )

    def norm(text):
        text = str(text).strip()
        tok = processor.tokenizer

        if hasattr(tok, "normalize"):
            return tok.normalize(text).strip()

        if hasattr(tok, "_normalize"):
            return tok._normalize(text).strip()

        raise RuntimeError(
            "Whisper tokenizer normalizer not found."
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = pred.label_ids.copy()

        label_ids[
            label_ids == -100
        ] = processor.tokenizer.pad_token_id

        pred_text = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )

        ref_text = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        preds = []
        refs = []

        for r, p in zip(ref_text, pred_text):
            r = norm(r)
            p = norm(p)

            if r:
                refs.append(r)
                preds.append(p)

        return {
            "wer": wer(refs, preds),
            "cer": cer(refs, preds),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=
            args.train_batch_size,

        per_device_eval_batch_size=
            args.eval_batch_size,

        gradient_accumulation_steps=
            args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,

        warmup_ratio=0.05,

        lr_scheduler_type="linear",

        fp16=True,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        predict_with_generate=True,
        generation_num_beams=5,
        generation_max_length=256,

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        save_total_limit=2,

        logging_steps=100,

        dataloader_num_workers=
            args.num_workers,

        remove_unused_columns=False,

        report_to=[],

        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,

        train_dataset=train_ds,
        eval_dataset=dev_ds,

        data_collator=collator,

        compute_metrics=compute_metrics,

        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    best_dir = output_dir / "best"

    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)

    info = {
        "best_checkpoint":
            trainer.state.best_model_checkpoint,
        "best_dev_wer":
            trainer.state.best_metric,
        "epochs":
            args.epochs,
        "learning_rate":
            args.learning_rate,
        "seed":
            args.seed,
    }

    with open(
        output_dir / "training_summary.json",
        "w",
    ) as f:
        json.dump(info, f, indent=2)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Best checkpoint :", info["best_checkpoint"])
    print("Best dev WER    :", info["best_dev_wer"])
    print("Saved model     :", best_dir)


if __name__ == "__main__":
    main()
