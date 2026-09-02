import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor

from train_mathspeech_projector_source_disjoint_hf import (
    DataCollatorSpeechSeq2SeqWithClap,
    MathSpeechDataset,
    WhisperSemanticASR,
    torch_load_compat,
)


def bootstrap_mean_ci(x, n_boot=10000, seed=42, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        boot_means[i] = sample.mean()
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return float(lo), float(hi)


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    whisper_name = train_args.get("whisper_name", "openai/whisper-base")
    clap_dim = int(ckpt.get("clap_dim", 512))

    model = WhisperSemanticASR(
        whisper_name=whisper_name,
        clap_dim=clap_dim,
        adapter_type=train_args.get("adapter_type", "gated"),
        pool_type=train_args.get("pool_type", "cls"),
        adapter_bottleneck=int(train_args.get("adapter_bottleneck", 256)),
        dropout=float(train_args.get("dropout", 0.1)),
        adapter_scale_init=float(train_args.get("adapter_scale_init", 0.01)),
        freeze_whisper=bool(train_args.get("freeze_whisper", True)),
    )

    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(
            f"{ckpt_path} does not contain 'model_state_dict'. "
            f"Available keys: {list(ckpt.keys())}"
        )

    msg = model.load_state_dict(state, strict=True)
    print("load_state_dict:", msg)
    model = model.to(device)
    model.eval()
    return model, ckpt, train_args


@torch.no_grad()
def extract_representation_scores(model, loader, device):
    rows = []
    all_z_orig = []
    all_z_new = []
    all_z_text = []
    all_sample_ids = []

    for batch in tqdm(loader, desc="representation analysis"):
        feats = batch["input_features"].to(device)
        z_text = batch["clap_emb"].to(device).float()

        # H_orig: frozen Whisper encoder output
        # H_new : projector/adaptor-applied encoder representation
        h_new, h_orig = model.encode_with_adapter(feats, return_original=True)

        # ----------------------------------------------------
        # Hidden representation preservation / drift
        # Measured directly in Whisper hidden space,
        # BEFORE AlignHead.
        # h_orig, h_new: [B, T, D]
        # ----------------------------------------------------
        h_orig_f = h_orig.float()
        h_new_f = h_new.float()

        # Higher = better preservation
        hidden_cosine = F.cosine_similarity(
            h_orig_f,
            h_new_f,
            dim=-1,
        ).mean(dim=-1)  # [B]

        # Lower = smaller drift
        hidden_mse = (
            (h_new_f - h_orig_f) ** 2
        ).mean(dim=(1, 2))  # [B]

        # Relative Frobenius L2 drift
        diff_norm = torch.sqrt(
            ((h_new_f - h_orig_f) ** 2).sum(dim=(1, 2))
        )
        orig_norm = torch.sqrt(
            (h_orig_f ** 2).sum(dim=(1, 2))
        ).clamp_min(1e-12)

        hidden_relative_l2 = diff_norm / orig_norm  # [B]

        # Current pool_type='cls' implementation means first encoder time step h[:, 0].
        pooled_orig = model.pooler(h_orig)
        pooled_new = model.pooler(h_new)

        # Use exactly the same trained AlignHead for before/after representations.
        z_orig = model.align_head(pooled_orig)
        z_new = model.align_head(pooled_new)

        z_orig_norm = F.normalize(z_orig.float(), dim=-1)
        z_new_norm = F.normalize(z_new.float(), dim=-1)
        z_text_norm = F.normalize(z_text.float(), dim=-1)

        s_orig = torch.sum(z_orig_norm * z_text_norm, dim=-1)
        s_new = torch.sum(z_new_norm * z_text_norm, dim=-1)
        delta = s_new - s_orig

        sample_ids = batch["sample_ids"]
        texts = batch["texts"]
        for i in range(len(sample_ids)):
            rows.append(
                {
                    "sample_id": int(sample_ids[i]),
                    "reference_text": texts[i],
                    "s_orig": float(s_orig[i].cpu()),
                    "s_new": float(s_new[i].cpu()),
                    "delta_s": float(delta[i].cpu()),
                    "improved": bool(delta[i].item() > 0),
                    "hidden_cosine": float(hidden_cosine[i].cpu()),
                    "hidden_mse": float(hidden_mse[i].cpu()),
                    "hidden_relative_l2": float(hidden_relative_l2[i].cpu()),
                }
            )

        all_z_orig.append(z_orig_norm.cpu())
        all_z_new.append(z_new_norm.cpu())
        all_z_text.append(z_text_norm.cpu())
        all_sample_ids.extend(int(x) for x in sample_ids)

    df = pd.DataFrame(rows)
    vectors = {
        "sample_ids": torch.tensor(all_sample_ids, dtype=torch.long),
        "z_orig": torch.cat(all_z_orig, dim=0),
        "z_new": torch.cat(all_z_new, dim=0),
        "z_text": torch.cat(all_z_text, dim=0),
    }
    return df, vectors


def make_paired_plot(df, output_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    rng = np.random.default_rng(42)

    for _, row in df.iterrows():
        x0 = 0 + rng.normal(0, 0.015)
        x1 = 1 + rng.normal(0, 0.015)
        ax.plot([x0, x1], [row["s_orig"], row["s_new"]], alpha=0.18, linewidth=0.7)
        ax.scatter([x0, x1], [row["s_orig"], row["s_new"]], s=7, alpha=0.25)

    means = [df["s_orig"].mean(), df["s_new"].mean()]
    ax.scatter([0, 1], means, s=80, marker="D", zorder=10, label="Mean")
    ax.plot([0, 1], means, linewidth=2.0, zorder=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before projector", "After projector"])
    ax.set_ylabel("Cosine similarity to Lecture-CLAP text target")
    ax.set_title("Paired Semantic Similarity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_delta_histogram(df, output_path):
    values = df["delta_s"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.hist(values, bins=25, alpha=0.85)
    ax.axvline(0, linestyle="--", linewidth=1.5, label="No change")
    ax.axvline(values.mean(), linestyle="-", linewidth=2.0, label=f"Mean ΔS = {values.mean():.4f}")
    ax.set_xlabel(r"$\Delta S = S_{\mathrm{new}} - S_{\mathrm{orig}}$")
    ax.set_ylabel("Number of samples")
    ax.set_title("Semantic Similarity Shift")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="MathSpeech validation/test manifest")
    ap.add_argument("--ckpt", required=True, help="Projector checkpoint (best.pt)")
    ap.add_argument(
        "--clap_emb_path",
        default="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("MATHSPEECH REPRESENTATION ANALYSIS")
    print("=" * 72)
    print("manifest :", args.manifest)
    print("ckpt     :", args.ckpt)
    print("CLAP     :", args.clap_emb_path)
    print("device   :", device)

    model, ckpt, train_args = load_model_from_checkpoint(args.ckpt, device)

    lambda_align = float(train_args.get("lambda_align", 0.0) or 0.0)
    align_loss_type = str(train_args.get("align_loss_type", "none"))
    semantic_alignment_supervised = (
        lambda_align > 0.0 and align_loss_type != "none"
    )

    print("-" * 72)
    print("checkpoint epoch       :", ckpt.get("epoch"))
    print("checkpoint valid WER   :", ckpt.get("valid_wer"))
    print("checkpoint best WER    :", ckpt.get("best_valid_wer"))
    print("LR                     :", train_args.get("lr"))
    print("lambda_align           :", train_args.get("lambda_align"))
    print("lambda_hidden          :", train_args.get("lambda_hidden"))
    print("adapter_type           :", train_args.get("adapter_type"))
    print("pool_type              :", train_args.get("pool_type"))
    print("freeze_whisper         :", train_args.get("freeze_whisper"))
    print("-" * 72)

    if not semantic_alignment_supervised:
        print(
            "[WARNING] This checkpoint did not train AlignHead with the CLAP "
            "alignment objective. Semantic alignment scores (S_orig, S_new, ΔS) "
            "are still computed for diagnostics but should NOT be interpreted "
            "as meaningful CLAP-space alignment. Use hidden-space preservation "
            "and drift metrics for this variant."
        )
        print("-" * 72)

    whisper_name = train_args.get("whisper_name", "openai/whisper-base")
    processor = WhisperProcessor.from_pretrained(whisper_name, language="English", task="transcribe")

    clap_embs = torch_load_compat(args.clap_emb_path, map_location="cpu").float()
    if clap_embs.ndim != 2:
        raise ValueError(f"Expected CLAP tensor [N,D], got {clap_embs.shape}")
    print("CLAP embeddings:", tuple(clap_embs.shape))

    dataset = MathSpeechDataset(args.manifest, processor, clap_embs)
    collator = DataCollatorSpeechSeq2SeqWithClap(
        processor,
        model.whisper.config.decoder_start_token_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    df, vectors = extract_representation_scores(model, loader, device)

    s_orig = df["s_orig"].to_numpy()
    s_new = df["s_new"].to_numpy()
    delta = df["delta_s"].to_numpy()

    hidden_cosine = df["hidden_cosine"].to_numpy()
    hidden_mse = df["hidden_mse"].to_numpy()
    hidden_relative_l2 = df["hidden_relative_l2"].to_numpy()

    ci_low, ci_high = bootstrap_mean_ci(
        delta,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    hidden_cos_ci_low, hidden_cos_ci_high = bootstrap_mean_ci(
        hidden_cosine,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    hidden_mse_ci_low, hidden_mse_ci_high = bootstrap_mean_ci(
        hidden_mse,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    hidden_l2_ci_low, hidden_l2_ci_high = bootstrap_mean_ci(
        hidden_relative_l2,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    summary = {
        "n": int(len(df)),
        "s_orig_mean": float(s_orig.mean()),
        "s_orig_std": float(s_orig.std(ddof=1)),
        "s_new_mean": float(s_new.mean()),
        "s_new_std": float(s_new.std(ddof=1)),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_std": float(delta.std(ddof=1)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "positive_ratio": float((delta > 0).mean()),
        "negative_ratio": float((delta < 0).mean()),
        "zero_ratio": float((delta == 0).mean()),
        "delta_mean_bootstrap_95ci": [ci_low, ci_high],
        "semantic_alignment_supervised": semantic_alignment_supervised,

        "hidden_cosine_mean": float(hidden_cosine.mean()),
        "hidden_cosine_std": float(hidden_cosine.std(ddof=1)),
        "hidden_cosine_bootstrap_95ci": [
            hidden_cos_ci_low,
            hidden_cos_ci_high,
        ],

        "hidden_mse_mean": float(hidden_mse.mean()),
        "hidden_mse_std": float(hidden_mse.std(ddof=1)),
        "hidden_mse_bootstrap_95ci": [
            hidden_mse_ci_low,
            hidden_mse_ci_high,
        ],

        "hidden_relative_l2_mean": float(hidden_relative_l2.mean()),
        "hidden_relative_l2_std": float(hidden_relative_l2.std(ddof=1)),
        "hidden_relative_l2_bootstrap_95ci": [
            hidden_l2_ci_low,
            hidden_l2_ci_high,
        ],

        "checkpoint": str(args.ckpt),
        "manifest": str(args.manifest),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_valid_wer": ckpt.get("valid_wer"),
        "checkpoint_best_valid_wer": ckpt.get("best_valid_wer"),
        "lr": train_args.get("lr"),
        "lambda_align": train_args.get("lambda_align"),
        "lambda_hidden": train_args.get("lambda_hidden"),
        "adapter_type": train_args.get("adapter_type"),
        "pool_type": train_args.get("pool_type"),
    }

    csv_path = outdir / "per_sample_representation.csv"
    summary_path = outdir / "representation_summary.json"
    vectors_path = outdir / "representation_vectors.pt"

    df.to_csv(csv_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(vectors, vectors_path)

    make_paired_plot(df, outdir / "paired_cosine_similarity.png")
    make_delta_histogram(df, outdir / "delta_cosine_distribution.png")

    print()
    print("=" * 72)
    print("REPRESENTATION ANALYSIS RESULT")
    print("=" * 72)
    print(f"N               : {len(df)}")
    print(f"S_orig mean     : {summary['s_orig_mean']:.6f}")
    print(f"S_new mean      : {summary['s_new_mean']:.6f}")
    print(f"Mean ΔS         : {summary['delta_mean']:+.6f}")
    print(f"Median ΔS       : {summary['delta_median']:+.6f}")
    print(f"Positive ratio  : {100 * summary['positive_ratio']:.2f}%")
    print(f"95% CI mean ΔS  : [{ci_low:+.6f}, {ci_high:+.6f}]")

    print()
    print("HIDDEN REPRESENTATION PRESERVATION")
    print("-" * 72)
    print(
        f"Hidden cosine   : {summary['hidden_cosine_mean']:.6f} "
        f"[{hidden_cos_ci_low:.6f}, {hidden_cos_ci_high:.6f}]"
    )
    print(
        f"Hidden MSE      : {summary['hidden_mse_mean']:.8f} "
        f"[{hidden_mse_ci_low:.8f}, {hidden_mse_ci_high:.8f}]"
    )
    print(
        f"Relative L2     : {summary['hidden_relative_l2_mean']:.6f} "
        f"[{hidden_l2_ci_low:.6f}, {hidden_l2_ci_high:.6f}]"
    )

    print("-" * 72)
    print("CSV             :", csv_path)
    print("summary         :", summary_path)
    print("vectors         :", vectors_path)
    print("paired plot     :", outdir / "paired_cosine_similarity.png")
    print("delta plot      :", outdir / "delta_cosine_distribution.png")


if __name__ == "__main__":
    main()
