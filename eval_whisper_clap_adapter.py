import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperProcessor

from mathspeech_utils import audio_path_for_index, load_audio_16k
from train_whisper_clap_adapter import WhisperSemanticASR


TAIL_ARTIFACT_PREFIXES = ("nd", "ndi", "ndx", "ndy", "ndt", "ndp")
TAIL_PHRASES = [
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
]


def clean_asr_tail(text: str) -> str:
    text = str(text).strip().replace(" ,", ",").replace(" .", ".")
    text = " ".join(text.split())
    words = text.split()
    if len(words) <= 4:
        return text.strip(" ,.")

    for i, word in enumerate(words):
        if i >= 3 and any(ord(ch) > 127 for ch in word):
            return " ".join(words[:i]).strip(" ,.")
        token = word.lower().strip(".,!?;:")
        if i >= 3 and any(token.startswith(prefix) for prefix in TAIL_ARTIFACT_PREFIXES):
            return " ".join(words[:i]).strip(" ,.")

    lowered = " " + text.lower()
    cut = None
    for phrase in TAIL_PHRASES:
        pos = lowered.find(phrase)
        if pos != -1 and len(text[: max(pos - 1, 0)].split()) >= 3:
            cut = pos if cut is None else min(cut, pos)
    if cut is not None:
        return text[: cut - 1].strip(" ,.")

    for n in [6, 5, 4, 3, 2]:
        if len(words) < n * 3:
            continue
        for i in range(len(words) - 3 * n + 1):
            p1 = [x.lower().strip(".,!?;:") for x in words[i : i + n]]
            p2 = [x.lower().strip(".,!?;:") for x in words[i + n : i + 2 * n]]
            p3 = [x.lower().strip(".,!?;:") for x in words[i + 2 * n : i + 3 * n]]
            if p1 == p2 == p3:
                return " ".join(words[: i + n]).strip(" ,.")

    return text.strip(" ,.")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--text_col", type=str, default="transcription")
    parser.add_argument("--audio_ext", type=str, default="mp3")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--eval_split", type=str, default="test", choices=["all", "train", "valid", "test"])
    parser.add_argument("--pred_col", type=str, default=None)
    parser.add_argument("--clean_tail", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    clap_dim = ckpt.get("clap_dim", 512)
    whisper_name = ckpt_args.get("whisper_name", args.whisper_name)

    processor = WhisperProcessor.from_pretrained(whisper_name, language="en", task="transcribe")
    model = WhisperSemanticASR(
        whisper_name=whisper_name,
        clap_dim=clap_dim,
        adapter_type=ckpt_args.get("adapter_type", "residual_mlp"),
        pool_type=ckpt_args.get("pool_type", "mean"),
        adapter_bottleneck=ckpt_args.get("adapter_bottleneck", 256),
        dropout=ckpt_args.get("dropout", 0.1),
        adapter_scale_init=ckpt_args.get("adapter_scale_init", 0.01),
        freeze_whisper=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    if args.eval_split == "all":
        eval_indices = list(range(len(df)))
    else:
        split_path = args.split_path or os.path.join(os.path.dirname(args.ckpt_path), "split_indices.pt")
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        eval_indices = split[f"{args.eval_split}_idx"]
        print("split_type:", split.get("split_type"))
        print("source_col:", split.get("source_col"))

    if args.pred_col is None:
        adapter_type = ckpt_args.get("adapter_type", "adapter")
        loss_type = ckpt_args.get("align_loss_type", "loss")
        args.pred_col = f"pred_{adapter_type}_{loss_type}_{args.eval_split}"

    print("eval_split:", args.eval_split)
    print("num eval samples:", len(eval_indices))
    print("prediction column:", args.pred_col)

    df[args.pred_col] = ""

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

                pred_ids = model.generate(
                    input_features=input_features,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                if args.clean_tail:
                    pred = clean_asr_tail(pred)
            except Exception as e:
                print(f"[error] index {real_idx + 1}: {e}")
                pred = ""

        df.loc[real_idx, args.pred_col] = pred
        print()
        print(f"[{real_idx + 1}]")
        print("GT  :", df[args.text_col].iloc[real_idx])
        print("PRED:", pred)

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    df.to_csv(args.save_csv, index=False)
    print("saved to:", args.save_csv)
    print("prediction column:", args.pred_col)


if __name__ == "__main__":
    main()
