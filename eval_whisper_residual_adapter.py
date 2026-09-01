import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor

from asr_dataset_utils import ASRCollator, ASRDataset, add_dataset_args, load_records_from_args
from train_whisper_residual_adapter import ResidualAdapterWhisper


@torch.no_grad()
def run_eval(model, loader, processor, device, max_new_tokens):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="eval"):
        input_features = batch["input_features"].to(device)
        pred_ids = model.generate(
            input_features=input_features,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        for uid, ref, pred, meta in zip(batch["uids"], batch["texts"], preds, batch["metadata"]):
            row = {"uid": uid, "transcription": ref, "prediction": pred}
            if isinstance(meta, dict):
                row.update({f"meta_{k}": v for k, v in meta.items()})
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--pred_col", type=str, default="pred_residual_adapter_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter_bottleneck", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--adapter_scale_init", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("dataset_type:", args.dataset_type)
    print("ckpt_path:", args.ckpt_path)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    adapter_bottleneck = args.adapter_bottleneck or int(train_args.get("adapter_bottleneck", 256))
    dropout = args.dropout if args.dropout is not None else float(train_args.get("dropout", 0.1))
    adapter_scale_init = args.adapter_scale_init if args.adapter_scale_init is not None else float(train_args.get("adapter_scale_init", 0.01))

    processor = WhisperProcessor.from_pretrained(args.whisper_name, language="en", task="transcribe")
    records = load_records_from_args(args, args.eval_split, save_dir=os.path.dirname(args.save_csv) or ".")
    loader = DataLoader(
        ASRDataset(records, processor, audio_ext=args.audio_ext),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ASRCollator(processor),
        pin_memory=True,
    )

    model = ResidualAdapterWhisper(
        whisper_name=args.whisper_name,
        adapter_bottleneck=adapter_bottleneck,
        dropout=dropout,
        adapter_scale_init=adapter_scale_init,
        freeze_whisper=True,
    ).to(device)
    model.adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    rows = run_eval(model, loader, processor, device, args.max_new_tokens)
    df = pd.DataFrame(rows)
    df[args.pred_col] = df.pop("prediction")
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    df.to_csv(args.save_csv, index=False)
    print("saved:", args.save_csv)
    print("num_eval:", len(df))


if __name__ == "__main__":
    main()
