import argparse
import inspect
import json
import math
from pathlib import Path

from jiwer import cer, wer
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from train_whisper_lora_controlled import ManifestDataset, WhisperCollator, normalize_with_tokenizer


def make_training_args(train_len, args, output_dir):
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    updates_per_epoch = math.ceil(train_len / max(1, args.train_batch_size * args.gradient_accumulation_steps))
    total_steps = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = math.ceil(total_steps * 0.05)
    kw = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        fp16=True,
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
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default="openai/whisper-base")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,out_proj,fc1,fc2")
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--generation_num_beams", type=int, default=5)
    p.add_argument("--generation_max_length", type=int, default=256)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_dev_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model, language="English", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(args.model)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    base_model.config.forced_decoder_ids = forced_decoder_ids
    base_model.generation_config.forced_decoder_ids = forced_decoder_ids
    base_model.generation_config.num_beams = args.generation_num_beams
    base_model.generation_config.max_length = args.generation_max_length

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=None,
        ),
    )

    non_lora_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if non_lora_trainable:
        raise RuntimeError("Unexpected non-LoRA trainable params:\n" + "\n".join(non_lora_trainable))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("CONTROLLED LORA-WHISPER COMPAT")
    print("targets:", target_modules)
    print("lr:", args.learning_rate, "epochs:", args.epochs, "trainable:", f"{trainable/1e6:.3f}M")
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
        pairs = []
        for ref, hyp in zip(ref_text, pred_text):
            r = normalize_with_tokenizer(processor.tokenizer, ref)
            h = normalize_with_tokenizer(processor.tokenizer, hyp)
            if r:
                pairs.append((r, h))
        return {"wer": wer([r for r, _ in pairs], [h for _, h in pairs]), "cer": cer([r for r, _ in pairs], [h for _, h in pairs])}

    targs = make_training_args(len(train_ds), args, outdir)
    trainer_kw = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kw["processing_class"] = processor
    elif "tokenizer" in trainer_params:
        trainer_kw["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kw)
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
        "trainable_percent": 100 * trainable / total,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "generation_num_beams": args.generation_num_beams,
        "generation_max_length": args.generation_max_length,
        "warmup_ratio_effective": 0.05,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_wer": trainer.state.best_metric,
        "seed": args.seed,
        "best_adapter_dir": str(best_adapter_dir),
        "merged_dir": str(merged_dir),
    }
    with open(outdir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("TRAINING COMPLETE | best dev WER:", trainer.state.best_metric, "| merged:", merged_dir)


if __name__ == "__main__":
    main()
