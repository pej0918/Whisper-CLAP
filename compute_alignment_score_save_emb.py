import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import whisper as openai_whisper
from transformers import WhisperProcessor

from train_whisper_projector_v2 import WhisperSemanticASR


# =========================================================
# Utils
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_clap_embeddings(path):
    """
    Flexible loader for CLAP text embeddings.

    Supported:
    - .pt tensor
    - .pt dict with one of:
        ["clap_embs", "text_embs", "embeddings", "tensor"]
    - .npy
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLAP embedding file not found: {path}")

    ext = os.path.splitext(path)[1]

    if ext == ".npy":
        arr = np.load(path)
        return torch.tensor(arr, dtype=torch.float32)

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.Tensor):
        return obj.float()

    if isinstance(obj, dict):
        for key in ["clap_embs", "text_embs", "embeddings", "tensor"]:
            if key in obj:
                val = obj[key]
                if isinstance(val, np.ndarray):
                    return torch.tensor(val, dtype=torch.float32)
                return val.float()

    raise ValueError(
        f"Unsupported CLAP embedding format in {path}. "
        f"Expected Tensor, .npy, or dict with keys like "
        f"['clap_embs', 'text_embs', 'embeddings', 'tensor']."
    )


# =========================================================
# Dataset
# =========================================================
class AlignmentDataset(Dataset):
    def __init__(self, df, indices, audio_dir, processor, clap_embs):
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.audio_dir = audio_dir
        self.processor = processor
        self.clap_embs = clap_embs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        text = str(row["transcription"])
        audio_path = os.path.join(self.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        audio = openai_whisper.load_audio(audio_path)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        item = {
            "input_features": input_features,
            "clap_emb": self.clap_embs[real_idx],
            "text": text,
            "real_idx": real_idx,
        }
        return item


class AlignmentCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = torch.stack([b["input_features"] for b in batch], dim=0)
        clap_emb = torch.stack([b["clap_emb"] for b in batch], dim=0)

        return {
            "input_features": input_features,
            "clap_emb": clap_emb,
            "texts": [b["text"] for b in batch],
            "real_indices": [b["real_idx"] for b in batch],
        }


# =========================================================
# Main
# =========================================================
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
    parser.add_argument(
        "--clap_emb_path",
        type=str,
        required=True,
        help="Path to precomputed GT transcription CLAP text embeddings (.pt or .npy)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to best.pt",
    )
    parser.add_argument(
        "--save_detail_csv",
        type=str,
        default=None,
        help="Per-sample alignment score CSV",
    )
    parser.add_argument(
        "--save_summary_csv",
        type=str,
        default=None,
        help="Summary CSV with mean/std/median/min/max",
    )
    parser.add_argument(
        "--save_emb_pt",
        type=str,
        default=None,
        help="Saved .pt containing audio_embs/text_embs/indices/texts",
    )
    parser.add_argument(
        "--whisper_name",
        type=str,
        default="openai/whisper-base",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["all", "train", "valid", "test"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # -----------------------------------------------------
    # Load dataframe / embeddings
    # -----------------------------------------------------
    df = pd.read_excel(args.excel_path)
    clap_embs = load_clap_embeddings(args.clap_emb_path)

    if len(clap_embs) != len(df):
        raise ValueError(
            f"CLAP embeddings length mismatch: len(clap_embs)={len(clap_embs)} "
            f"vs len(df)={len(df)}"
        )

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    clap_dim = ckpt.get("clap_dim", clap_embs.shape[-1])

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
    model = model.to(device)
    model.eval()

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------
    if args.eval_split == "all":
        eval_indices = list(range(len(df)))
    else:
        split_path = os.path.join(os.path.dirname(args.ckpt_path), "split_indices.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"split_indices.pt not found: {split_path}")

        split = torch.load(split_path, map_location="cpu", weights_only=False)
        eval_indices = split[f"{args.eval_split}_idx"]

    print("eval_split:", args.eval_split)
    print("num eval samples:", len(eval_indices))

    # -----------------------------------------------------
    # Dataset / Loader
    # -----------------------------------------------------
    dataset = AlignmentDataset(
        df=df,
        indices=eval_indices,
        audio_dir=args.audio_dir,
        processor=processor,
        clap_embs=clap_embs,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=AlignmentCollator(processor),
        pin_memory=True,
    )

    # -----------------------------------------------------
    # Save paths
    # -----------------------------------------------------
    save_dir = os.path.dirname(args.ckpt_path)

    if args.save_detail_csv is None:
        args.save_detail_csv = os.path.join(
            save_dir, f"alignment_score_{args.eval_split}.csv"
        )
    if args.save_summary_csv is None:
        args.save_summary_csv = os.path.join(
            save_dir, f"alignment_score_{args.eval_split}_summary.csv"
        )
    if args.save_emb_pt is None:
        args.save_emb_pt = os.path.join(
            save_dir, f"alignment_embeddings_{args.eval_split}.pt"
        )

    os.makedirs(os.path.dirname(args.save_detail_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_summary_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_emb_pt), exist_ok=True)

    # -----------------------------------------------------
    # Compute alignment
    # -----------------------------------------------------
    all_scores = []
    all_rows = []

    all_audio_embs = []
    all_text_embs = []
    all_indices = []
    all_texts = []

    for batch in tqdm(loader):
        input_features = batch["input_features"].to(device)
        target_emb = batch["clap_emb"].to(device)

        # audio-side projected embedding
        h_adapted = model.encode_with_adapter(input_features)   # [B, T, D]
        pooled = model.pooler(h_adapted)                       # [B, D]
        audio_emb = model.align_head(pooled)                   # [B, clap_dim]

        # cosine similarity
        audio_emb_norm = F.normalize(audio_emb, dim=-1)
        target_emb_norm = F.normalize(target_emb, dim=-1)
        sim = F.cosine_similarity(audio_emb_norm, target_emb_norm, dim=-1)  # [B]

        # save rows
        sim_cpu = sim.detach().cpu()
        audio_cpu = audio_emb.detach().cpu()
        text_cpu = target_emb.detach().cpu()

        for i in range(len(batch["real_indices"])):
            row = {
                "real_idx": int(batch["real_indices"][i]),
                "text": batch["texts"][i],
                "alignment_score": float(sim_cpu[i].item()),
            }
            all_rows.append(row)

        all_scores.extend(sim_cpu.tolist())
        all_audio_embs.append(audio_cpu)
        all_text_embs.append(text_cpu)
        all_indices.extend(batch["real_indices"])
        all_texts.extend(batch["texts"])

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    method_name = os.path.basename(os.path.dirname(args.ckpt_path))

    scores_np = np.array(all_scores, dtype=np.float64)

    summary = {
        "method": method_name,
        "num_eval": len(scores_np),
        "alignment_mean": float(scores_np.mean()),
        "alignment_std": float(scores_np.std()),
        "alignment_median": float(np.median(scores_np)),
        "alignment_min": float(scores_np.min()),
        "alignment_max": float(scores_np.max()),
    }

    # -----------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------
    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(args.save_detail_csv, index=False)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.save_summary_csv, index=False)

    # -----------------------------------------------------
    # Save embedding .pt
    # -----------------------------------------------------
    save_obj = {
        "method": method_name,
        "eval_split": args.eval_split,
        "indices": all_indices,
        "texts": all_texts,
        "audio_embs": torch.cat(all_audio_embs, dim=0),   # [N, C]
        "text_embs": torch.cat(all_text_embs, dim=0),     # [N, C]
        "scores": torch.tensor(all_scores, dtype=torch.float32),
    }
    torch.save(save_obj, args.save_emb_pt)

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------
    print("\n===== Alignment Score Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("saved detail :", args.save_detail_csv)
    print("saved summary:", args.save_summary_csv)
    print("saved emb pt :", args.save_emb_pt)


if __name__ == "__main__":
    main()