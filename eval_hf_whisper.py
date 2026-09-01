import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from mathspeech_utils import audio_path_for_index, load_audio_16k


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--model_name", type=str, default="openai/whisper-base")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional fine-tuned Whisper checkpoint.")
    parser.add_argument("--text_col", type=str, default="transcription")
    parser.add_argument("--pred_col", type=str, default=None)
    parser.add_argument("--audio_ext", type=str, default="mp3")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)

    if args.eval_split == "all":
        eval_indices = list(range(len(df)))
    else:
        split = torch.load(args.split_path, map_location="cpu", weights_only=False)
        eval_indices = split[f"{args.eval_split}_idx"]
        print("split_type:", split.get("split_type"))
        print("source_col:", split.get("source_col"))

    processor = WhisperProcessor.from_pretrained(args.model_name, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        print("loaded checkpoint:", args.ckpt_path)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    model.to(device)
    model.eval()

    if args.pred_col is None:
        prefix = "pred_hf_whisper_base" if args.ckpt_path is None else "pred_hf_whisper_ft"
        args.pred_col = f"{prefix}_{args.eval_split}"

    df[args.pred_col] = ""

    print("eval_split:", args.eval_split)
    print("num_eval:", len(eval_indices))
    print("prediction column:", args.pred_col)

    use_amp = args.fp16 and torch.cuda.is_available()

    for real_idx in tqdm(eval_indices):
        audio_path = audio_path_for_index(args.audio_dir, real_idx, ext=args.audio_ext)

        if not os.path.exists(audio_path):
            print("[missing]", audio_path)
            pred = ""
        else:
            try:
                audio = load_audio_16k(audio_path)
                input_features = processor.feature_extractor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_ids = model.generate(
                        input_features=input_features,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        do_sample=False,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )

                pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                print(f"[error] index {real_idx + 1}: {e}")
                pred = ""

        df.loc[real_idx, args.pred_col] = pred
        print()
        print(f"[{real_idx + 1}]")
        print("GT  :", df[args.text_col].iloc[real_idx])
        print("PRED:", pred)

    df.to_csv(args.save_csv, index=False)
    print("saved to:", args.save_csv)
    print("prediction column:", args.pred_col)


if __name__ == "__main__":
    main()
