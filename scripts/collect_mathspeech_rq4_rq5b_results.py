import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def infer_lr(dirname, train):
    if train:
        for k in ("learning_rate", "lr"):
            if train.get(k) is not None:
                return train.get(k)
        if isinstance(train.get("args"), dict):
            for k in ("learning_rate", "lr"):
                if train["args"].get(k) is not None:
                    return train["args"].get(k)
    if "_lr" in dirname:
        return dirname.split("_lr", 1)[1].split("_", 1)[0]
    return None


def classify_dir(name):
    if name.startswith("rq4_ce_only_lr"):
        return "RQ4", "CE only", "CE", "gated b256"
    if name.startswith("rq4_ce_hidden_lr"):
        return "RQ4", "CE + Hidden", "CE+Hidden", "gated b256"
    if name.startswith("rq4_ce_align_lr"):
        return "RQ4", "CE + Align", "CE+Align", "gated b256"
    if name.startswith("rq5b_lora_r7_nooutproj_lr"):
        return "RQ5-B", "LoRA r=7", None, "r=7, alpha=7, -out_proj"
    if name.startswith("rq5b_residual_b128_lr"):
        return "RQ5-B", "Residual Adapter b=128", None, "bottleneck=128"
    return None


def get_best_valid_wer(train):
    if not train:
        return None
    for key in ("best_dev_wer", "best_valid_wer"):
        if train.get(key) is not None:
            return train.get(key)
    return None


def get_trainable_params(train):
    if not train:
        return None
    if train.get("trainable_params") is not None:
        return int(train["trainable_params"])
    if train.get("trainable_params_M") is not None:
        return int(round(float(train["trainable_params_M"]) * 1_000_000))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    root = Path(args.output_dir)
    rows = []

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = classify_dir(d.name)
        if cls is None:
            continue
        rq, method, loss_cfg, config = cls
        train = load_json(d / "training_summary.json")
        test = load_json(d / "test_summary_beam5.json")
        if test is None:
            continue

        params = get_trainable_params(train)
        rows.append({
            "RQ": rq,
            "Method": method,
            "Loss Config": loss_cfg,
            "Configuration": config,
            "LR": infer_lr(d.name, train),
            "Beam": 5,
            "Best Valid WER": get_best_valid_wer(train),
            "WER": test.get("wer"),
            "CER": test.get("cer"),
            "RTF": test.get("rtf"),
            "Trainable Params": params,
            "Trainable Params (M)": None if params is None else params / 1_000_000,
            "Result Dir": str(d),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        order = {
            "CE only": 0,
            "CE + Hidden": 1,
            "CE + Align": 2,
            "LoRA r=7": 3,
            "Residual Adapter b=128": 4,
        }
        df["_order"] = df["Method"].map(order).fillna(99)
        df = df.sort_values(["RQ", "_order", "LR"]).drop(columns="_order")

    full_csv = root / "rq4_rq5b_all_results.csv"
    full_md = root / "rq4_rq5b_all_results.md"
    df.to_csv(full_csv, index=False)
    full_md.write_text(df.to_markdown(index=False) + "\n" if not df.empty else "No results.\n")

    # Select LR independently per method using validation WER only.
    selected_rows = []
    if not df.empty:
        for (_, method), g in df.groupby(["RQ", "Method"], sort=False):
            g2 = g.dropna(subset=["Best Valid WER"])
            if g2.empty:
                continue
            selected_rows.append(g2.loc[g2["Best Valid WER"].astype(float).idxmin()])
    best = pd.DataFrame(selected_rows)
    if not best.empty:
        best = best.reset_index(drop=True)

    best_csv = root / "rq4_rq5b_best_valid_lr_results.csv"
    best_md = root / "rq4_rq5b_best_valid_lr_results.md"
    best.to_csv(best_csv, index=False)
    best_md.write_text(best.to_markdown(index=False) + "\n" if not best.empty else "No selected results.\n")

    print("=" * 120)
    print("ALL RESULTS")
    print(df.to_markdown(index=False) if not df.empty else "No results")
    print("=" * 120)
    print("VALIDATION-SELECTED LR RESULTS")
    print(best.to_markdown(index=False) if not best.empty else "No selected results")
    print("=" * 120)
    print("saved:", full_csv)
    print("saved:", full_md)
    print("saved:", best_csv)
    print("saved:", best_md)


if __name__ == "__main__":
    main()
