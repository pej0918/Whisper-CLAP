import os
import argparse
import torch
import pandas as pd

from train_whisper_projector_v2 import WhisperSemanticASR


def count_params(model):
    total = 0
    trainable = 0

    whisper_total = 0
    whisper_trainable = 0

    projector_total = 0
    projector_trainable = 0

    align_total = 0
    align_trainable = 0

    other_total = 0
    other_trainable = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

        lname = name.lower()

        if lname.startswith("whisper."):
            whisper_total += n
            if p.requires_grad:
                whisper_trainable += n

        elif any(k in lname for k in ["adapter", "projector", "align_head", "pool", "gate"]):
            projector_total += n
            if p.requires_grad:
                projector_trainable += n

        elif "align" in lname:
            align_total += n
            if p.requires_grad:
                align_trainable += n

        else:
            other_total += n
            if p.requires_grad:
                other_trainable += n

    return {
        "total_params": total,
        "trainable_params": trainable,
        "whisper_total": whisper_total,
        "whisper_trainable": whisper_trainable,
        "projector_total": projector_total,
        "projector_trainable": projector_trainable,
        "align_total": align_total,
        "align_trainable": align_trainable,
        "other_total": other_total,
        "other_trainable": other_trainable,
        "trainable_ratio_percent": 100.0 * trainable / total if total > 0 else 0.0,
    }


def load_metric_summary(exp_dir):
    metric_path = os.path.join(exp_dir, "metrics_test_summary.csv")
    if not os.path.exists(metric_path):
        return {}

    df = pd.read_csv(metric_path)
    row = df.iloc[0].to_dict()

    out = {}
    for key in ["wer", "cer", "mathterm_recall", "overbias", "tailrate"]:
        if key in row:
            out[key.upper() if key in ["wer", "cer"] else key] = row[key]

    # 혹시 column 이름이 대문자인 경우 대비
    for key in ["WER", "CER", "MathTerm Recall", "OverBias", "TailRate"]:
        if key in row:
            out[key] = row[key]

    return out


def load_alignment_summary(exp_dir):
    align_path = os.path.join(exp_dir, "alignment_score_test_summary.csv")
    if not os.path.exists(align_path):
        return {}

    df = pd.read_csv(align_path)
    row = df.iloc[0].to_dict()

    if "alignment_mean" in row:
        return {"Align Score": row["alignment_mean"]}
    if "align_score" in row:
        return {"Align Score": row["align_score"]}

    return {}


def pretty_millions(x):
    return round(x / 1_000_000, 4)


def infer_method_name(exp):
    mapping = {
        "whisper_base_ft_only": "Whisper fine-tuning only",
        "v5_arch_residual_mlp_ce": "Projector CE-only / Residual MLP",
        "v5_arch_gated_ce": "Projector CE-only / Gated",
        "v5_arch_bottleneck_ce": "Projector CE-only / Bottleneck",
        "v5_arch_linear_residual_ce": "Projector CE-only / Linear Residual",
        "v5_arch_conv1d_ce": "Projector CE-only / Conv1D",
        "v5_loss_residual_cosine": "Ours / Residual MLP / cosine",
        "v5_pool_gated_cls_cosine": "Ours, best ASR / Gated cls",
        "v5_pool_gated_attn_cosine": "Ours, best alignment / Gated attn",
        "v5_pool_gated_mean_cosine": "Ours / Gated mean",
        "v5_pos_post_encoder_cosine": "Ours / post-encoder",
        "v5_pos_encoder_layer3_cosine": "Ours / encoder layer 3",
        "v5_pos_encoder_layer6_cosine": "Ours / encoder layer 6",
        "v5_pos_encoder_layer9_cosine": "Ours / encoder layer 9",
        "v5_pos_both_layer6_cosine": "Ours / both layer 6",
    }
    return mapping.get(exp, exp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "whisper_base_ft_only",
            "v5_arch_gated_ce",
            "v5_pool_gated_cls_cosine",
            "v5_pool_gated_attn_cosine",
            "v5_loss_residual_cosine",
        ],
    )
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    args = parser.parse_args()

    rows = []

    for exp in args.experiments:
        exp_dir = os.path.join(args.base_dir, exp)
        ckpt_path = os.path.join(exp_dir, "best.pt")

        print("=" * 80)
        print(exp)
        print("ckpt:", ckpt_path)

        if not os.path.exists(ckpt_path):
            print("[skip] missing ckpt")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        clap_dim = ckpt.get("clap_dim", 512)

        whisper_name = ckpt_args.get("whisper_name", args.whisper_name)

        model = WhisperSemanticASR(
            whisper_name=whisper_name,
            clap_dim=clap_dim,
            adapter_type=ckpt_args.get("adapter_type", "residual_mlp"),
            pool_type=ckpt_args.get("pool_type", "mean"),
            adapter_bottleneck=ckpt_args.get("adapter_bottleneck", 256),
            dropout=ckpt_args.get("dropout", 0.1),
            adapter_scale_init=ckpt_args.get("adapter_scale_init", 0.01),
            freeze_whisper=ckpt_args.get("freeze_whisper", True),
        )

        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"],
            strict=False,
        )

        if len(missing) > 0:
            print("[warning] missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("[warning] unexpected keys:", unexpected[:10])

        stats = count_params(model)
        metric_stats = load_metric_summary(exp_dir)
        align_stats = load_alignment_summary(exp_dir)

        row = {
            "Experiment": exp,
            "Method": infer_method_name(exp),
            "Adapter": ckpt_args.get("adapter_type", "-"),
            "Pool": ckpt_args.get("pool_type", "-"),
            "Position": ckpt_args.get("projector_position", ckpt_args.get("position", "post-encoder")),
            "Align Loss": ckpt_args.get("align_loss_type", "-"),
            "Total Params": stats["total_params"],
            "Trainable Params": stats["trainable_params"],
            "Trainable Params (M)": pretty_millions(stats["trainable_params"]),
            "Total Params (M)": pretty_millions(stats["total_params"]),
            "Trainable Ratio (%)": round(stats["trainable_ratio_percent"], 4),
            "Whisper Trainable": stats["whisper_trainable"],
            "Projector Trainable": stats["projector_trainable"],
            "Align Head Trainable": stats["align_trainable"],
            "Other Trainable": stats["other_trainable"],
        }

        row.update(metric_stats)
        row.update(align_stats)

        rows.append(row)

        print("total params:", stats["total_params"])
        print("trainable params:", stats["trainable_params"])
        print("trainable params M:", pretty_millions(stats["trainable_params"]))
        print("trainable ratio:", round(stats["trainable_ratio_percent"], 4), "%")
        print("projector trainable:", stats["projector_trainable"])
        print("whisper trainable:", stats["whisper_trainable"])

    df = pd.DataFrame(rows)

    preferred_cols = [
        "Experiment",
        "Method",
        "Adapter",
        "Pool",
        "Position",
        "Align Loss",
        "Trainable Params",
        "Trainable Params (M)",
        "Total Params (M)",
        "Trainable Ratio (%)",
        "Projector Trainable",
        "Whisper Trainable",
        "WER",
        "CER",
        "mathterm_recall",
        "overbias",
        "tailrate",
        "Align Score",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]

    print("\n===== PARAMETER SUMMARY =====")
    print(df.to_string(index=False))

    if args.save_csv is None:
        args.save_csv = os.path.join(args.base_dir, "param_count_summary.csv")

    df.to_csv(args.save_csv, index=False)
    print("\nsaved to:", args.save_csv)


if __name__ == "__main__":
    main()