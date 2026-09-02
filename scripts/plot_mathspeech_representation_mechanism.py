import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run(path):
    path = Path(path)

    df = pd.read_csv(
        path / "per_sample_representation.csv"
    )

    with open(path / "representation_summary.json") as f:
        summary = json.load(f)

    return df, summary


def asymmetric_yerr(means, cis):
    lower = [
        mean - ci[0]
        for mean, ci in zip(means, cis)
    ]
    upper = [
        ci[1] - mean
        for mean, ci in zip(means, cis)
    ]

    return np.array([lower, upper])


def add_point_values(
    ax,
    xs,
    ys,
    offset_frac=0.05,
    fmt=".4f",
):
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * offset_frac

    for x, y in zip(xs, ys):
        ax.text(
            x,
            y + offset,
            format(y, fmt),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ce_align_dir",
        required=True,
    )
    parser.add_argument(
        "--full_dir",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=22,
    )

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ce_df, ce = load_run(args.ce_align_dir)
    full_df, full = load_run(args.full_dir)

    # ---------------------------------------------------------
    # Sanity check
    # ---------------------------------------------------------
    if not np.array_equal(
        ce_df["sample_id"].to_numpy(),
        full_df["sample_id"].to_numpy(),
    ):
        raise ValueError(
            "CE+Align and Full do not use the same samples/order."
        )

    ce_delta = ce_df["delta_s"].to_numpy()
    full_delta = full_df["delta_s"].to_numpy()

    # =========================================================
    # Figure
    # =========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.0),
        gridspec_kw={
            "width_ratios": [1.55, 1.0, 1.0],
        },
    )

    # =========================================================
    # (a) Semantic alignment shift
    # =========================================================

    ax = axes[0]

    global_min = min(
        ce_delta.min(),
        full_delta.min(),
    )
    global_max = max(
        ce_delta.max(),
        full_delta.max(),
    )

    bins = np.linspace(
        global_min,
        global_max,
        args.bins + 1,
    )

    _, _, ce_patches = ax.hist(
        ce_delta,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="CE + Align",
    )

    _, _, full_patches = ax.hist(
        full_delta,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="Full (Ours)",
    )

    # Use histogram colors for their corresponding mean lines.
    ce_color = ce_patches[0].get_edgecolor()
    full_color = full_patches[0].get_edgecolor()

    # No-change reference
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.3,
        color="black",
        label="No change",
    )

    # Mean lines
    ax.axvline(
        ce["delta_mean"],
        linestyle=":",
        linewidth=1.7,
        color=ce_color,
        label="_nolegend_",
    )

    ax.axvline(
        full["delta_mean"],
        linestyle=":",
        linewidth=1.7,
        color=full_color,
        label="_nolegend_",
    )

    # Mean shift is indicated by the method-colored dotted lines.
    # Exact values are reported in the table/caption.

    ax.set_xlabel(
        r"$\Delta$ semantic alignment"
        "\n"
        r"$(S_{\mathrm{adapted}}-S_{\mathrm{original}})$"
    )

    ax.set_ylabel("Density")
    ax.set_title("(a) Semantic alignment shift")

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper right",
    )

    # =========================================================
    # Common x positions for (b), (c)
    # =========================================================

    names = [
        "CE + Align",
        "Full\n(Ours)",
    ]

    x = np.arange(2)

    # =========================================================
    # (b) Hidden-state preservation
    # Point estimate + 95% bootstrap CI
    # =========================================================

    ax = axes[1]

    hidden_cos = np.array([
        ce["hidden_cosine_mean"],
        full["hidden_cosine_mean"],
    ])

    hidden_cos_ci = [
        ce["hidden_cosine_bootstrap_95ci"],
        full["hidden_cosine_bootstrap_95ci"],
    ]

    hidden_cos_err = asymmetric_yerr(
        hidden_cos,
        hidden_cos_ci,
    )

    # Keep method colors consistent across all panels.
    ax.errorbar(
        x[0],
        hidden_cos[0],
        yerr=hidden_cos_err[:, 0].reshape(2, 1),
        fmt="o",
        markersize=8,
        capsize=5,
        elinewidth=1.6,
        capthick=1.6,
        linestyle="none",
        color=ce_color,
    )

    ax.errorbar(
        x[1],
        hidden_cos[1],
        yerr=hidden_cos_err[:, 1].reshape(2, 1),
        fmt="o",
        markersize=8,
        capsize=5,
        elinewidth=1.6,
        capthick=1.6,
        linestyle="none",
        color=full_color,
    )

    # Dynamic y range suitable for point estimates
    low = min(
        ci[0]
        for ci in hidden_cos_ci
    )
    high = max(
        ci[1]
        for ci in hidden_cos_ci
    )

    span = high - low
    pad = max(
        span * 0.35,
        0.005,
    )

    ax.set_ylim(
        low - pad,
        min(1.005, high + pad * 1.7),
    )

    # Place values slightly outward so they do not collide
    # with the points or plot boundaries.
    ax.annotate(
        f"{hidden_cos[0]:.4f}",
        xy=(x[0], hidden_cos[0]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    ax.annotate(
        f"{hidden_cos[1]:.4f}",
        xy=(x[1], hidden_cos[1]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.12, 1.12)

    ax.set_ylabel(
        r"$\cos(H_{\mathrm{orig}}, H_{\mathrm{adapted}})$"
    )

    ax.set_title(
        "(b) Hidden-state preservation ↑"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    # =========================================================
    # (c) Relative representation drift
    # Point estimate + 95% bootstrap CI
    # =========================================================

    ax = axes[2]

    rel_l2 = np.array([
        ce["hidden_relative_l2_mean"],
        full["hidden_relative_l2_mean"],
    ])

    rel_l2_ci = [
        ce["hidden_relative_l2_bootstrap_95ci"],
        full["hidden_relative_l2_bootstrap_95ci"],
    ]

    rel_l2_err = asymmetric_yerr(
        rel_l2,
        rel_l2_ci,
    )

    # Keep method colors consistent with panels (a) and (b).
    ax.errorbar(
        x[0],
        rel_l2[0],
        yerr=rel_l2_err[:, 0].reshape(2, 1),
        fmt="o",
        markersize=8,
        capsize=5,
        elinewidth=1.6,
        capthick=1.6,
        linestyle="none",
        color=ce_color,
    )

    ax.errorbar(
        x[1],
        rel_l2[1],
        yerr=rel_l2_err[:, 1].reshape(2, 1),
        fmt="o",
        markersize=8,
        capsize=5,
        elinewidth=1.6,
        capthick=1.6,
        linestyle="none",
        color=full_color,
    )

    low = min(
        ci[0]
        for ci in rel_l2_ci
    )
    high = max(
        ci[1]
        for ci in rel_l2_ci
    )

    span = high - low
    pad = max(
        span * 0.25,
        0.015,
    )

    ax.set_ylim(
        max(0, low - pad),
        high + pad * 1.6,
    )

    ax.annotate(
        f"{rel_l2[0]:.4f}",
        xy=(x[0], rel_l2[0]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    ax.annotate(
        f"{rel_l2[1]:.4f}",
        xy=(x[1], rel_l2[1]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.12, 1.12)

    ax.set_ylabel(
        r"$\|H_{\mathrm{adapted}}-H_{\mathrm{orig}}\|_2"
        r"/\|H_{\mathrm{orig}}\|_2$"
    )

    ax.set_title(
        "(c) Representation drift ↓"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    # =========================================================
    # Final layout
    # =========================================================

    fig.tight_layout(
        w_pad=2.3,
    )

    png = outdir / "representation_mechanism_final.png"
    pdf = outdir / "representation_mechanism_final.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================
    # Paper-ready quantitative summary
    # =========================================================

    mse_reduction = (
        1
        - full["hidden_mse_mean"]
        / ce["hidden_mse_mean"]
    ) * 100

    l2_reduction = (
        1
        - full["hidden_relative_l2_mean"]
        / ce["hidden_relative_l2_mean"]
    ) * 100

    print("=" * 78)
    print("REPRESENTATION MECHANISM ANALYSIS")
    print("=" * 78)

    print(
        f"{'Variant':<16}"
        f"{'Δ Align':>10}"
        f"{'Hidden Cos ↑':>15}"
        f"{'MSE ↓':>13}"
        f"{'Rel L2 ↓':>13}"
    )

    print("-" * 78)

    for name, s in [
        ("CE + Align", ce),
        ("Full (Ours)", full),
    ]:
        print(
            f"{name:<16}"
            f"{s['delta_mean']:>+10.4f}"
            f"{s['hidden_cosine_mean']:>15.4f}"
            f"{s['hidden_mse_mean']:>13.4f}"
            f"{s['hidden_relative_l2_mean']:>13.4f}"
        )

    print()

    print(
        f"Hidden MSE reduction (Full vs CE+Align): "
        f"{mse_reduction:.1f}%"
    )

    print(
        f"Relative L2 drift reduction             : "
        f"{l2_reduction:.1f}%"
    )

    print()
    print("Saved:")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
