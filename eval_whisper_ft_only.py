import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm

import whisper as openai_whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def clean_asr_tail(text):
    text = str(text).strip()
    text = text.replace(" ,", ",").replace(" .", ".")
    text = " ".join(text.split())

    words = text.split()
    if len(words) <= 4:
        return text.strip(" ,.")

    for i, w in enumerate(words):
        if i >= 3 and any(ord(ch) > 127 for ch in w):
            return " ".join(words[:i]).strip(" ,.")

    cut_prefixes = ["nd", "ndi", "ndx", "ndy", "ndt", "ndp", "ida", "ire", "irex", "icoosine"]
    for i, w in enumerate(words):
        ww = w.lower().strip(".,!?;:")
        if i >= 3 and any(ww.startswith(p) for p in cut_prefixes):
            return " ".join(words[:i]).strip(" ,.")

    tail_phrases = [
        " more than",
        " more ",
        " of r",
        " of l",
        " of y",
        " of pi",
        " going to be",
        " it is",
        " it's",
        " the same as",
        " right here",
        " okay",
        " that's a big",
    ]

    lowered = " " + text.lower()
    best_cut = None

    for phrase in tail_phrases:
        pos = lowered.find(phrase)
        if pos != -1:
            prefix = text[:max(pos - 1, 0)].strip()
            if len(prefix.split()) >= 3:
                if best_cut is None or pos < best_cut:
                    best_cut = pos

    if best_cut is not None:
        return text[:best_cut - 1].strip(" ,.")

    words = text.split()
    for n in [6, 5, 4, 3, 2]:
        if len(words) < n * 3:
            continue
        for i in range(len(words) - 3 * n + 1):
            p1 = [x.lower().strip(".,!?;:") for x in words[i:i+n]]
            p2 = [x.lower().strip(".,!?;:") for x in words[i+n:i+2*n]]
            p3 = [x.lower().strip(".,!?;:") for x in words[i+2*n:i+3*n]]
            if p1 == p2 == p3:
                return " ".join(words[:i+n]).strip(" ,.")

    return text.strip(" ,.")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)

    parser.add_argument("--eval_split", type=str, default="test", choices=["all", "train", "valid", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--pred_col", type=str, default="pred_whisper_ft_only")
    parser.add_argument("--clean_tail", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    whisper_name = ckpt_args.get("whisper_name", "openai/whisper-base")

    df = pd.read_excel(args.excel_path)

    processor = WhisperProcessor.from_pretrained(
        whisper_name,
        language="en",
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(whisper_name)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe",
    )
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    if args.eval_split == "all":
        eval_indices = list(range(len(df)))
    else:
        split_path = os.path.join(os.path.dirname(args.ckpt_path), "split_indices.pt")
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        eval_indices = split[f"{args.eval_split}_idx"]

    print("eval_split:", args.eval_split)
    print("num eval samples:", len(eval_indices))

    df[args.pred_col] = ""

    for real_idx in tqdm(eval_indices):
        audio_path = os.path.join(args.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            print("[missing]", audio_path)
            pred = ""
        else:
            audio = openai_whisper.load_audio(audio_path)
            input_features = processor.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device)

            pred_ids = model.generate(
                input_features=input_features,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

            pred = processor.batch_decode(
                pred_ids,
                skip_special_tokens=True,
            )[0].strip()

            if args.clean_tail:
                pred = clean_asr_tail(pred)

        df.loc[real_idx, args.pred_col] = pred

        print()
        print(f"[{real_idx + 1}]")
        print("GT  :", df["transcription"].iloc[real_idx])
        print("PRED:", pred)

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    df.to_csv(args.save_csv, index=False)

    print("saved to:", args.save_csv)


if __name__ == "__main__":
    main()