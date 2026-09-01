import argparse
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from jiwer import cer, wer
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, set_seed

TARGET_SR = 16000


def load_segment(path, start, end):
    start, end = float(start), float(end)
    if end <= start:
        raise ValueError(f"Invalid segment: {path}, {start}, {end}")
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        f.seek(int(round(start * sr)))
        wav = f.read(frames=int(round((end - start) * sr)), dtype="float32", always_2d=True)
    if wav.shape[0] == 0:
        raise RuntimeError(f"Empty audio: {path} [{start}, {end}]")
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = AF.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy()
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
        if (self.df["duration"].astype(float) > 30.0).any():
            raise RuntimeError("Found segment >30 sec")
        print(f"{manifest}: {len(self.df)} samples, {self.df['duration'].astype(float).sum()/3600:.2f} h")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return {"audio": load_segment(r["audio_path"], r["start"], r["end"]), "text": str(r["text_spoken"])}


class WhisperCollator:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        batch = self.processor.feature_extractor([x["audio"] for x in features], sampling_rate=TARGET_SR, return_tensors="pt")
        lb = self.processor.tokenizer([x["text"] for x in features], padding=True, return_tensors="pt")
        labels = lb["input_ids"].masked_fill(lb["attention_mask"].ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def norm(tok, text):
    text = str(text).strip()
    if hasattr(tok, "normalize"): return tok.normalize(text).strip()
    if hasattr(tok, "_normalize"): return tok._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found")


def make_training_args(train_len, args, output_dir):
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    updates_per_epoch = math.ceil(train_len / max(1, args.train_batch_size * args.gradient_accumulation_steps))
    total_steps = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = math.ceil(total_steps * 0.05)
    kw = dict(
        output_dir=str(output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        lr_scheduler_type="linear", fp16=True, save_strategy="epoch", predict_with_generate=True,
        generation_num_beams=args.generation_num_beams, generation_max_length=args.generation_max_length,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        save_total_limit=2, logging_steps=100, dataloader_num_workers=args.num_workers,
        remove_unused_columns=False, report_to=[], seed=args.seed, data_seed=args.seed,
    )
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = 0.05
    elif "warmup_steps" in params:
        kw["warmup_steps"] = warmup_steps
    if "evaluation_strategy" in params:
        kw["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        kw["eval_strategy"] = "epoch"
    kept = {k: v for k, v in kw.items() if k in params}
    print("transformers-compatible warmup: 5% ->", warmup_steps, "steps; total steps:", total_steps)
    print("training args keys:", sorted(kept))
    return Seq2SeqTrainingArguments(**kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="openai/whisper-base"); ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--learning_rate", type=float, default=1e-5); ap.add_argument("--train_batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=16); ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4); ap.add_argument("--generation_num_beams", type=int, default=5)
    ap.add_argument("--generation_max_length", type=int, default=256); ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_dev_samples", type=int, default=None); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); set_seed(args.seed)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    processor = WhisperProcessor.from_pretrained(args.model, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.config.forced_decoder_ids = forced_ids; model.generation_config.forced_decoder_ids = forced_ids
    model.generation_config.num_beams = args.generation_num_beams; model.generation_config.max_length = args.generation_max_length
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("FULL FINE-TUNING COMPAT | trainable:", f"{trainable/1e6:.3f}M", "lr:", args.learning_rate, "epochs:", args.epochs)
    train_ds, dev_ds = ManifestDataset(args.train, args.max_train_samples), ManifestDataset(args.dev, args.max_dev_samples)
    collator = WhisperCollator(processor, model.config.decoder_start_token_id)

    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = pred.label_ids.copy(); label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(norm(processor.tokenizer, r), norm(processor.tokenizer, p)) for r, p in zip(refs, preds)]
        pairs = [(r, p) for r, p in pairs if r]
        return {"wer": wer([r for r, _ in pairs], [p for _, p in pairs]), "cer": cer([r for r, _ in pairs], [p for _, p in pairs])}

    targs = make_training_args(len(train_ds), args, output_dir)
    trainer_kw = dict(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=collator, compute_metrics=compute_metrics)
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params: trainer_kw["processing_class"] = processor
    elif "tokenizer" in trainer_params: trainer_kw["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kw); trainer.train()
    best_dir = output_dir / "best"; trainer.save_model(best_dir); processor.save_pretrained(best_dir)
    info = {"base_model": args.model, "method": "Full Fine-tuning", "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_dev_wer": trainer.state.best_metric, "epochs": args.epochs, "learning_rate": args.learning_rate,
            "trainable_params": trainable, "trainable_params_M": trainable/1e6, "generation_num_beams": args.generation_num_beams,
            "generation_max_length": args.generation_max_length, "warmup_ratio_effective": 0.05, "seed": args.seed}
    with open(output_dir / "training_summary.json", "w") as f: json.dump(info, f, indent=2)
    print("TRAINING COMPLETE | best dev WER:", trainer.state.best_metric, "| saved:", best_dir)

if __name__ == "__main__": main()
