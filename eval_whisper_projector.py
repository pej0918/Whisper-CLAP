import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm

import whisper as openai_whisper

from transformers import WhisperProcessor

from train_whisper_projector import WhisperSemanticASR


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel_path",
        type=str,
        default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/data1/eunju/datasets/mathspeech/dataset",
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="all",
        choices=["all", "train", "valid", "test"],
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    clap_dim = ckpt.get("clap_dim", 512)

    processor = WhisperProcessor.from_pretrained(
        args.whisper_name,
        language="en",
        task="transcribe",
    )

    model = WhisperSemanticASR(
        whisper_name=args.whisper_name,
        clap_dim=clap_dim,
        adapter_bottleneck=ckpt_args.get("adapter_bottleneck", 256),
        freeze_whisper=True,
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe",
    )
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    # split 선택
    if args.eval_split == "all":
        eval_indices = list(range(len(df)))
    else:
        save_dir = os.path.dirname(args.ckpt_path)
        split_path = os.path.join(save_dir, "split_indices.pt")

        if not os.path.exists(split_path):
            raise FileNotFoundError(f"split file not found: {split_path}")

        split = torch.load(split_path, map_location="cpu", weights_only=False)

        if args.eval_split == "train":
            eval_indices = split["train_idx"]
        elif args.eval_split == "valid":
            eval_indices = split["valid_idx"]
        else:
            eval_indices = split["test_idx"]

    print("eval_split:", args.eval_split)
    print("num eval samples:", len(eval_indices))

    pred_dict = {}

    for real_idx in tqdm(eval_indices):
        audio_path = os.path.join(args.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            print("[missing]", audio_path)
            pred_dict[real_idx] = ""
            continue

        try:
            audio = openai_whisper.load_audio(audio_path)

            input_features = processor.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device)

            # pred_ids = model.generate(
            #     input_features=input_features,
            #     max_new_tokens=args.max_new_tokens,
            # )
            pred_ids = model.generate(
                input_features=input_features,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

            pred = processor.batch_decode(
                pred_ids,
                skip_special_tokens=True,
            )[0].strip()

            pred_dict[real_idx] = pred

            print()
            print(f"[{real_idx + 1}]")
            print("GT  :", df['transcription'].iloc[real_idx])
            print("PRED:", pred)

        except Exception as e:
            print(f"[error] index {real_idx + 1}: {e}")
            pred_dict[real_idx] = ""

    col_name = f"projector_pred_{args.eval_split}"
    df[col_name] = ""

    for idx, pred in pred_dict.items():
        df.loc[idx, col_name] = pred

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    df.to_csv(args.save_csv, index=False)

    print("saved to:", args.save_csv)


if __name__ == "__main__":
    main()
