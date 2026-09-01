# plot_asr_alignment_tradeoff.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


EXP_INFO = {
    # main baselines
    "whisper_base_ft_only": {
        "label": "FT only",
        "group": "baseline",
    },

    # projector ablation
    "v5_arch_residual_mlp_ce": {
        "label": "Residual CE",
        "group": "CE-only",
    },
    "v5_arch_gated_ce": {
        "label": "Gated CE",
        "group": "CE-only",
    },
    "v5_arch_bottleneck_ce": {
        "label": "Bottleneck CE",
        "group": "CE-only",
    },
    "v5_arch_linear_residual_ce": {
        "label": "Linear Residual CE",
        "group": "CE-only",
    },
    "v5_arch_conv1d_ce": {
        "label": "Conv1D CE",
        "group": "CE-only",
    },

    # loss ablation
    "v5_loss_residual_ce": {
        "label": "Residual CE",
        "group": "loss",
    },
    "v5_loss_residual_cosine": {
        "label": "Residual cosine",
        "group": "loss",
    },
    "v5_loss_residual_mse": {
        "label": "Residual MSE",
        "group": "loss",
    },
    "v5_loss_residual_clip": {
        "label": "Residual CLIP",
        "group": "loss",
    },
    "v5_loss_residual_cosine_clip": {
        "label": "Residual cos+CLIP",
        "group": "loss",
    },
    "v5_loss_residual_cosine_mse": {
        "label": "Residual cos+MSE",
        "group": "loss",
    },
    "v5_loss_residual_all": {
        "label": "Residual all",
        "group": "loss",
    },

    # position ablation
    "v5_pos_post_encoder_cosine": {
        "label": "Post-enc",
        "group": "position",
    },
    "v5_pos_encoder_layer3_cosine": {
        "label": "Layer 3",
        "group": "position",
    },
    "v5_pos_encoder_layer6_cosine": {
        "label": "Layer 6",
        "group": "position",
    },
    "v5_pos_encoder_layer9_cosine": {
        "label": "Layer 9",
        "group": "position",
    },
    "v5_pos_both_layer6_cosine": {
        "label": "Both L6",
        "group": "position",
    },

    # pooling ablation
    "v5_pool_gated_mean_cosine": {
        "label": "Gated mean",
        "group": "pooling",
    },
    "v5_pool_gated_attn_cosine": {
        "label": "Gated attn",
        "group": "pooling",
    },
    "v5_pool_gated_cls_cosine": {
        "label": "Gated cls",
        "group": "pooling",
    },
}


def read_metric_value(csv_path, keys):
    """
    metrics_test_summary.csv 형식이 다음 둘 중 하나여도 읽히게 처리:
    1) columns: metric, value
    2) columns: WER, CER, MathTerm Recall, ...
    """
    df = pd.read_csv(csv_path)

    # case 1: metric-value long format
    lower_cols = [c.lower() for c in df.columns]
    if "metric" in lower_cols and "value" in lower_cols:
        metric_col = df.columns[lower_cols.index("metric")]
        value_col = df.columns[lower_cols.index("value")]

        metric_map = {
            str(row[metric_col]).strip().lower(): row[value_col]
            for _, row in df.iterrows()
        }

        for k in keys:
            kk = k.lower()
            if kk in metric_map:
                return float(metric_map[kk])

        return None

    # case 2: one-row wide format
    for k in keys:
        for col in df.columns:
            if col.strip().lower() == k.lower():
                return float(df[col].iloc[0])

    return None


def load_experiment(base_dir, exp_name):
    exp_dir = os.path.join(base_dir, exp_name)

    metric_path = os.path.join(exp_dir, "metrics_test_summary.csv")
    align_path = os.path.join(exp_dir, "alignment_score_test_summary.csv")

    if not os.path.exists(metric_path):
        print(f"[skip] missing metrics: {metric_path}")
        return None

    wer = read_metric_value(metric_path, ["wer", "WER"])
    cer = read_metric_value(metric_path, ["cer", "CER"])
    math_recall = read_metric_value(
        metric_path,
        ["mathterm_recall", "MathTerm Recall", "math_term_recall"],
    )
    overbias = read_metric_value(
        metric_path,
        ["overbias", "OverBias", "over_bias"],
    )
    tailrate = read_metric_value(
        metric_path,
        ["tailrate", "TailRate", "tail_rate"],
    )

    align = None
    if os.path.exists(align_path):
        align = read_metric_value(
            align_path,
            ["alignment_mean", "align_score", "Align Score"],
        )

        # 혹시 summary csv가 wide format인데 alignment_mean column으로 저장된 경우
        if align is None:
            adf = pd.read_csv(align_path)
            if "alignment_mean" in adf.columns:
                align = float(adf["alignment_mean"].iloc[0])

    info = EXP_INFO.get(exp_name, {"label": exp_name, "group": "other"})

    return {
        "exp": exp_name,
        "label": info["label"],
        "group": info["group"],
        "WER": wer,
        "CER": cer,
        "MathTerm Recall": math_recall,
        "OverBias": overbias,
        "TailRate": tailrate,
        "Align Score": align,
    }


def plot_tradeoff(df, y_metric, save_path):
    plot_df = df.dropna(subset=["Align Score", y_metric]).copy()

    plt.figure(figsize=(9, 6))

    groups = plot_df["group"].unique()

    for group in groups:
        sub = plot_df[plot_df["group"] == group]
        plt.scatter(
            sub["Align Score"],
            sub[y_metric],
            s=90,
            alpha=0.8,
            label=group,
        )

    # 핵심 포인트 annotation
    highlight_names = {
        "v5_pool_gated_cls_cosine": "Best ASR\nGated-cls",
        "v5_pool_gated_attn_cosine": "Best Align\nGated-attn",
        "v5_arch_gated_ce": "Gated CE",
        "v5_loss_residual_cosine": "Residual cosine",
    }

    for _, row in plot_df.iterrows():
        exp = row["exp"]
        if exp in highlight_names:
            plt.annotate(
                highlight_names[exp],
                xy=(row["Align Score"], row[y_metric]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
            )

    plt.xlabel("Alignment Score ↑")
    plt.ylabel(f"{y_metric} ↓")
    plt.title(f"ASR–Alignment trade-off ({y_metric})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/figures",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    for exp_name in EXP_INFO.keys():
        row = load_experiment(args.base_dir, exp_name)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    summary_path = os.path.join(args.save_dir, "asr_alignment_tradeoff_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"saved summary: {summary_path}")

    print()
    print(df[["exp", "label", "group", "WER", "CER", "MathTerm Recall", "OverBias", "TailRate", "Align Score"]])

    plot_tradeoff(
        df,
        y_metric="WER",
        save_path=os.path.join(args.save_dir, "tradeoff_align_vs_wer.png"),
    )

    plot_tradeoff(
        df,
        y_metric="CER",
        save_path=os.path.join(args.save_dir, "tradeoff_align_vs_cer.png"),
    )


if __name__ == "__main__":
    main()