import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from jiwer import cer, wer
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

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
        n_long = int((self.df["duration"].astype(float) > 30.0).sum())
        if n_long > 0:
            raise RuntimeError(f"{n_long} segments > 30 sec")
        print(f"{manifest}: {len(self.df)} samples, {self.df['duration'].astype(float).sum()/3600:.2f} h")

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
        batch = self.processor.feature_extractor(audios, sampling_rate=TARGET_SR, return_tensors="pt")
        labels_batch = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        if (
            labels.ndim == 2
            and labels.shape[1] > 0
            and self.decoder_start_token_id is not None
            and (labels[:, 0] == self.decoder_start_token_id).all().item()
        ):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def normalize_with_tokenizer(tokenizer, text):
    text = str(text).strip()
    if hasattr(tokenizer, "normalize"):
        return tokenizer.normalize(text).strip()
    if hasattr(tokenizer, "_normalize"):
        return tokenizer._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--train_batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--generation_num_beams", type=int, default=5)
    ap.add_argument("--generation_max_length", type=int, default=256)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_dev_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.config.forced_decoder_ids = forced_ids
    model.generation_config.forced_decoder_ids = forced_ids

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print("FULL FINE-TUNING COMPAT")
    print("trainable params:", f"{trainable:,}", f"({trainable/1e6:.3f}M)")
    print("all params      :", f"{total:,}")
    print("learning rate   :", args.learning_rate)
    print("epochs          :", args.epochs)

    train_ds = ManifestDataset(args.train, args.max_train_samples)
    dev_ds = ManifestDataset(args.dev, args.max_dev_samples)
    collator = WhisperCollator(processor, model.config.decoder_start_token_id)

    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_text = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        refs, hyps = [], []
        for ref, hyp in zip(ref_text, pred_text):
            ref = normalize_with_tokenizer(processor.tokenizer, ref)
            hyp = normalize_with_tokenizer(processor.tokenizer, hyp)
            if ref:
                refs.append(ref)
                hyps.append(hyp)
        return {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_num_beams=args.generation_num_beams,
        generation_max_length=args.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=100,
        dataloader_num_workers=args.num_workers,
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
        "base_model": args.model,
        "method": "Full Fine-tuning",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_wer": trainer.state.best_metric,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "trainable_params": trainable,
        "trainable_params_M": trainable / 1e6,
        "generation_num_beams": args.generation_num_beams,
        "generation_max_length": args.generation_max_length,
        "seed": args.seed,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(info, f, indent=2)
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("Best dev WER:", trainer.state.best_metric)
    print("Saved model :", best_dir)


if __name__ == "__main__":
    main()
