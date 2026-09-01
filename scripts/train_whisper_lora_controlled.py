import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from jiwer import cer, wer
from peft import LoraConfig, get_peft_model
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
        audio_batch = self.processor.feature_extractor(
            audios, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        label_batch = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = label_batch["input_ids"].masked_fill(label_batch["attention_mask"].ne(1), -100)
        if (
            labels.ndim == 2
            and labels.shape[1] > 0
            and self.decoder_start_token_id is not None
            and (labels[:, 0] == self.decoder_start_token_id).all().item()
        ):
            labels = labels[:, 1:]
        return {"input_features": audio_batch["input_features"], "labels": labels}


def normalize_with_tokenizer(tokenizer, text):
    text = str(text).strip()
    if hasattr(tokenizer, "normalize"):
        return tokenizer.normalize(text).strip()
    if hasattr(tokenizer, "_normalize"):
        return tokenizer._normalize(text).strip()
    raise RuntimeError("Whisper tokenizer normalizer not found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="openai/whisper-base")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,fc1,fc2")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--generation_num_beams", type=int, default=5)
    parser.add_argument("--generation_max_length", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model, language="English", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(args.model)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    base_model.config.forced_decoder_ids = forced_decoder_ids
    base_model.generation_config.forced_decoder_ids = forced_decoder_ids

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    non_lora_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if non_lora_trainable:
        raise RuntimeError("Unexpected non-LoRA trainable params:\n" + "\n".join(non_lora_trainable))

    print("=" * 70)
    print("CONTROLLED LORA-WHISPER")
    print("base model      :", args.model)
    print("rank            :", args.rank)
    print("lora alpha      :", args.lora_alpha)
    print("target modules  :", target_modules)
    print("learning rate   :", args.learning_rate)
    print("epochs          :", args.epochs)
    print("trainable params:", f"{trainable:,}", f"({trainable/1e6:.3f}M)")
    print("trainable %     :", f"{100*trainable/total:.3f}%")
    model.print_trainable_parameters()

    train_ds = ManifestDataset(args.train, args.max_train_samples)
    dev_ds = ManifestDataset(args.dev, args.max_dev_samples)
    collator = WhisperCollator(processor, base_model.config.decoder_start_token_id)

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
        output_dir=str(outdir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        optim="adamw_torch",
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

    best_adapter_dir = outdir / "best_adapter"
    trainer.model.save_pretrained(best_adapter_dir)
    processor.save_pretrained(best_adapter_dir)

    merged_model = trainer.model.merge_and_unload()
    merged_model.config.forced_decoder_ids = forced_decoder_ids
    merged_model.generation_config.forced_decoder_ids = forced_decoder_ids
    merged_dir = outdir / "merged"
    merged_model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)

    summary = {
        "base_model": args.model,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "trainable_params": trainable,
        "trainable_params_M": trainable / 1e6,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "generation_num_beams": args.generation_num_beams,
        "generation_max_length": args.generation_max_length,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_wer": trainer.state.best_metric,
        "seed": args.seed,
        "best_adapter_dir": str(best_adapter_dir),
        "merged_dir": str(merged_dir),
    }
    with open(outdir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("best checkpoint:", trainer.state.best_model_checkpoint)
    print("best dev WER   :", trainer.state.best_metric)
    print("adapter        :", best_adapter_dir)
    print("merged model   :", merged_dir)


if __name__ == "__main__":
    main()
