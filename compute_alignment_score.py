import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

import whisper as openai_whisper
from transformers import WhisperProcessor

from train_whisper_projector_v2 import WhisperSemanticASR


def load_clap_embeddings(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        if "embeddings" in obj:
            obj = obj["embeddings"]
        elif "text_emb" in obj:
            obj = obj["text_emb"]
        elif "clap_emb" in obj:
            obj = obj["clap_emb"]
        else:
            raise ValueError(f"Unknown CLAP embedding dict keys: {obj.keys()}")

    return obj.float()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clap_emb_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt")
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="test", choices=["all", "train", "valid", "test"])
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    clap_dim = ckpt.get("clap_dim", 512)

    whisper_name = ckpt_args.get("whisper_name", args.whisper_name)

    processor = WhisperProcessor.from_pretrained(
        whisper_name,
        language="en",
        task="transcribe",
    )

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

    clap_embs = load_clap_embeddings(args.clap_emb_path)

    if args.eval_split == "all":
        eval_indices = list(range(len(df)))
    else:
        split_path = os.path.join(os.path.dirname(args.ckpt_path), "split_indices.pt")
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        eval_indices = split[f"{args.eval_split}_idx"]

    print("eval_split:", args.eval_split)
    print("num eval samples:", len(eval_indices))

    rows = []

    for real_idx in tqdm(eval_indices):
        audio_path = os.path.join(args.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            print("[missing]", audio_path)
            continue

        audio = openai_whisper.load_audio(audio_path)

        input_features = processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)

        h_adapted = model.encode_with_adapter(input_features)
        pooled = model.pooler(h_adapted)
        z = model.align_head(pooled)

        target = clap_embs[real_idx].unsqueeze(0).to(device)

        z = F.normalize(z.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)

        cosine = F.cosine_similarity(z, target, dim=-1).item()

        rows.append({
            "index": real_idx,
            "transcription": df["transcription"].iloc[real_idx],
            "alignment_cosine": cosine,
        })

    out_df = pd.DataFrame(rows)

    summary = {
        "method": os.path.basename(os.path.dirname(args.ckpt_path)),
        "num_eval": len(out_df),
        "alignment_mean": out_df["alignment_cosine"].mean(),
        "alignment_std": out_df["alignment_cosine"].std(),
        "alignment_median": out_df["alignment_cosine"].median(),
        "alignment_min": out_df["alignment_cosine"].min(),
        "alignment_max": out_df["alignment_cosine"].max(),
    }

    print("\n===== Alignment Score Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)

    out_df.to_csv(args.save_csv, index=False)

    summary_path = args.save_csv.replace(".csv", "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("saved detail:", args.save_csv)
    print("saved summary:", summary_path)


if __name__ == "__main__":
    main()