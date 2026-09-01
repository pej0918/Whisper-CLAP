import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import whisper


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
    parser.add_argument(
        "--split_path",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v4_align_residual_mlp_cosine_hidden/split_indices.pt",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/openai_whisper_base/result_ASR_test.csv",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "all"],
    )

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

    print("eval_split:", args.eval_split)
    print("num_eval:", len(eval_indices))

    model = whisper.load_model("base").to(device)
    model.eval()

    pred_dict = {}

    for real_idx in tqdm(eval_indices):
        audio_path = os.path.join(args.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            print("[missing]", audio_path)
            pred_dict[real_idx] = ""
            continue

        try:
            result = model.transcribe(
                audio_path,
                language="en",
                fp16=torch.cuda.is_available(),
            )

            pred = result["text"].strip()
            pred_dict[real_idx] = pred

            print()
            print(f"[{real_idx + 1}]")
            print("GT  :", df["transcription"].iloc[real_idx])
            print("PRED:", pred)

        except Exception as e:
            print(f"[error] index {real_idx + 1}: {e}")
            pred_dict[real_idx] = ""

    pred_col = f"openai_whisper_base_pred_{args.eval_split}"
    df[pred_col] = ""

    for idx, pred in pred_dict.items():
        df.loc[idx, pred_col] = pred

    df.to_csv(args.save_csv, index=False)

    print("saved to:", args.save_csv)
    print("prediction column:", pred_col)


if __name__ == "__main__":
    main()
