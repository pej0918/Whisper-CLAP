import argparse
import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

from mathspeech_utils import (
    audio_path_for_index,
    load_audio_16k,
    make_or_load_source_aware_split,
    set_seed,
)


class MathSpeechWhisperDataset(Dataset):
    def __init__(self, df, indices, audio_dir, processor, text_col="transcription", audio_ext="mp3"):
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.audio_dir = audio_dir
        self.processor = processor
        self.text_col = text_col
        self.audio_ext = audio_ext

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        real_idx = self.indices[item]
        audio_path = audio_path_for_index(self.audio_dir, real_idx, ext=self.audio_ext)
        audio = load_audio_16k(audio_path)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        text = str(self.df[self.text_col].iloc[real_idx]).strip()
        labels = self.processor.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
            "real_idx": real_idx,
        }


class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = torch.stack([x["input_features"] for x in batch], dim=0)
        labels = pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        real_idx = torch.tensor([x["real_idx"] for x in batch], dtype=torch.long)
        return {"input_features": input_features, "labels": labels, "real_idx": real_idx}


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum_steps, use_amp):
    model.train()
    total_loss = 0.0
    step_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train")):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(input_features=input_features, labels=labels)
            loss = out.loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        step_count += 1

    return total_loss / max(step_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    step_count = 0

    for batch in tqdm(loader, desc="valid"):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(input_features=input_features, labels=labels)

        total_loss += out.loss.item()
        step_count += 1

    return total_loss / max(step_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--text_col", type=str, default="transcription")
    parser.add_argument("--audio_ext", type=str, default="mp3")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--source_col", type=str, default=None)
    parser.add_argument("--force_new_split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)
    print("num samples:", len(df))

    split = make_or_load_source_aware_split(
        df=df,
        save_dir=args.save_dir,
        split_path=args.split_path,
        source_col=args.source_col,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        force_new=args.force_new_split,
    )

    processor = WhisperProcessor.from_pretrained(args.whisper_name, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_name)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    if args.freeze_decoder:
        for p in model.model.decoder.parameters():
            p.requires_grad = False

    model.to(device)

    train_dataset = MathSpeechWhisperDataset(
        df=df,
        indices=split["train_idx"],
        audio_dir=args.audio_dir,
        processor=processor,
        text_col=args.text_col,
        audio_ext=args.audio_ext,
    )
    valid_dataset = MathSpeechWhisperDataset(
        df=df,
        indices=split["valid_idx"],
        audio_dir=args.audio_dir,
        processor=processor,
        text_col=args.text_col,
        audio_ext=args.audio_ext,
    )

    collator = WhisperCollator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs // max(args.grad_accum_steps, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_valid = float("inf")
    log_rows = []
    log_path = os.path.join(args.save_dir, "train_log.csv")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=use_amp,
        )
        valid_loss = evaluate_loss(model=model, loader=valid_loader, device=device, use_amp=use_amp)

        print(f"train_loss: {train_loss:.6f}")
        print(f"valid_loss: {valid_loss:.6f}")
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "valid_loss": valid_loss,
            "split_type": split.get("split_type"),
            "source_col": split.get("source_col"),
        }
        last_path = os.path.join(args.save_dir, "last.pt")
        torch.save(ckpt, last_path)

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            ckpt["best_valid_loss"] = best_valid
            torch.save(ckpt, best_path)
            print("saved best:", best_path)

    print("best_valid_loss:", best_valid)


if __name__ == "__main__":
    main()
