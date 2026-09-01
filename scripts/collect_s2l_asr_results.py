import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_protocol(root):
    path = Path(root) / "protocol.txt"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def classify_dir(name):
    if name == "whisper_base":
        return "Whisper-base"
    if name.startswith("fullft_lr"):
        return "Full Fine-tuning"
    if name.startswith("clap_fullft_lr"):
        return "CLAP-guided Full Fine-tuning"
    if name.startswith("ours_lr"):
        return "Ours"
    if name.startswith("lora_whisper_lr") and name.endswith("_outproj"):
        return "LoRA-Whisper +out_proj"
    if name.startswith("lora_whisper_lr") and name.endswith("_nooutproj"):
        return "LoRA-Whisper -out_proj"
    if name.startswith("residual_b256_lr"):
        return "KAUST-style Residual Adapter"
    return None


def infer_lr(dirname, train):
    if train:
        for key in ("learning_rate", "lr"):
            if train.get(key) is not None:
                return train.get(key)
        if isinstance(train.get("args"), dict):
            for key in ("learning_rate", "lr"):
                if train["args"].get(key) is not None:
                    return train["args"].get(key)
    m = re.search(r"_lr([^_]+)", dirname)
    return m.group(1) if m else None


def infer_trainable_params(method, train):
    if method == "Whisper-base":
        return 0
    if not train:
        return None
    if train.get("trainable_params") is not None:
        return int(train["trainable_params"])
    if train.get("trainable_params_M") is not None:
        return int(round(float(train["trainable_params_M"]) * 1_000_000))
    return None


def get_best_valid_wer(train):
    if not train:
        return None
    for key in ("best_dev_wer", "best_valid_wer"):
        if train.get(key) is not None:
            return float(train[key])
    return None


def write_table(df, csv_path, md_path):
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("No completed results found.\n" if df.empty else df.to_markdown(index=False) + "\n")
    print("saved csv:", csv_path)
    print("saved md :", md_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--best_csv", default=None)
    ap.add_argument("--best_md", default=None)
    args = ap.parse_args()

    root = Path(args.output_dir)
    protocol = load_protocol(root)
    rows = []

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        method = classify_dir(d.name)
        if method is None:
            continue
        train = load_json(d / "training_summary.json")
        lr = infer_lr(d.name, train)
        params = infer_trainable_params(method, train)
        valid_wer = get_best_valid_wer(train)

        for test_dir in sorted(p for p in d.glob("test_*") if p.is_dir()):
            test_source = test_dir.name.replace("test_", "", 1)
            for summary_path in sorted(test_dir.glob("summary_beam*.json")):
                m = re.search(r"summary_beam(\d+)\.json$", summary_path.name)
                if not m:
                    continue
                test = load_json(summary_path)
                if not test:
                    continue
                beam = int(m.group(1))

                target_modules = None
                if train:
                    target_modules = train.get("target_modules")
                    if target_modules is None and isinstance(train.get("args"), dict):
                        target_modules = train["args"].get("target_modules")

                freeze_whisper = test.get("freeze_whisper")
                if freeze_whisper is None and train and isinstance(train.get("args"), dict):
                    freeze_whisper = train["args"].get("freeze_whisper")

                rows.append({
                    "Task": protocol.get("TASK"),
                    "Train Source": protocol.get("TRAIN_SOURCE"),
                    "Valid Source": protocol.get("VALID_SOURCE"),
                    "Test Source": test_source,
                    "Method": method,
                    "LR": lr,
                    "Beam": beam,
                    "WER": test.get("wer"),
                    "CER": test.get("cer"),
                    "RTF": test.get("rtf"),
                    "Num Eval": test.get("samples", test.get("valid_samples")),
                    "Trainable Params": params,
                    "Trainable Params (M)": None if params is None else params / 1_000_000,
                    "Best Valid WER": valid_wer,
                    "Freeze Whisper": freeze_whisper,
                    "LoRA Targets": target_modules,
                    "Result Dir": str(d),
                })

    df = pd.DataFrame(rows)
    method_order = {
        "Whisper-base": 0,
        "Full Fine-tuning": 1,
        "CLAP-guided Full Fine-tuning": 2,
        "LoRA-Whisper +out_proj": 3,
        "LoRA-Whisper -out_proj": 4,
        "KAUST-style Residual Adapter": 5,
        "Ours": 6,
    }
    source_order = {"mix": 0, "h": 1, "a": 2}

    if not df.empty:
        df["_method_order"] = df["Method"].map(method_order).fillna(99)
        df["_source_order"] = df["Test Source"].map(source_order).fillna(99)
        df = df.sort_values(
            ["_method_order", "LR", "_source_order", "Beam"], na_position="first"
        ).drop(columns=["_method_order", "_source_order"])

    print("=" * 180)
    print("No completed S2L summaries found under " + str(root) if df.empty else df.to_markdown(index=False))
    print("=" * 180)

    write_table(
        df,
        Path(args.csv) if args.csv else root / "s2l_asr_results.csv",
        Path(args.md) if args.md else root / "s2l_asr_results.md",
    )

    # Main-paper helper: select LR using validation WER only, then retain
    # every requested test source and beam for that selected LR.
    selected_parts = []
    if not df.empty:
        for method, g in df.groupby("Method", sort=False):
            if method == "Whisper-base":
                selected_parts.append(g)
                continue
            candidates = g.dropna(subset=["Best Valid WER"])
            if candidates.empty:
                continue
            lr_scores = candidates.groupby("LR", dropna=False)["Best Valid WER"].min()
            best_lr = lr_scores.idxmin()
            selected_parts.append(g[g["LR"].astype(str) == str(best_lr)])

    best_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else df.iloc[0:0].copy()
    if not best_df.empty:
        best_df["_method_order"] = best_df["Method"].map(method_order).fillna(99)
        best_df["_source_order"] = best_df["Test Source"].map(source_order).fillna(99)
        best_df = best_df.sort_values(["_method_order", "_source_order", "Beam"]).drop(
            columns=["_method_order", "_source_order"]
        )

    write_table(
        best_df,
        Path(args.best_csv) if args.best_csv else root / "s2l_best_valid_lr_results.csv",
        Path(args.best_md) if args.best_md else root / "s2l_best_valid_lr_results.md",
    )


if __name__ == "__main__":
    main()
