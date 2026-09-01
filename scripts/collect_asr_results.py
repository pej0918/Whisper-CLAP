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
    if "_lr" in dirname:
        return dirname.split("_lr", 1)[1].split("_", 1)[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    root = Path(args.output_dir)
    rows = []

    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        method = classify_dir(d.name)
        if method is None:
            continue
        train = load_json(d / "training_summary.json")
        lr = infer_lr(d.name, train)

        for beam in (1, 5):
            test = load_json(d / f"test_summary_beam{beam}.json")
            if not test:
                continue

            target_modules = None
            if train:
                target_modules = train.get("target_modules")
                if target_modules is None and isinstance(train.get("args"), dict):
                    target_modules = train["args"].get("target_modules")

            freeze_whisper = test.get("freeze_whisper")
            if freeze_whisper is None and train and isinstance(train.get("args"), dict):
                freeze_whisper = train["args"].get("freeze_whisper")

            trainable_params = None
            if train:
                trainable_params = train.get("trainable_params")
                if trainable_params is None:
                    trainable_params = train.get("trainable_params_M")

            best_valid = None
            if train:
                best_valid = train.get("best_dev_wer", train.get("best_valid_wer"))
                if best_valid is None:
                    best_valid = train.get("best_valid_loss")

            rows.append({
                "Method": method,
                "LR": lr,
                "Beam": beam,
                "WER": test.get("wer"),
                "CER": test.get("cer"),
                "RTF": test.get("rtf"),
                "Num Eval": test.get("samples", test.get("valid_samples")),
                "Trainable Params": trainable_params,
                "Best Valid Metric": best_valid,
                "Freeze Whisper": freeze_whisper,
                "LoRA Targets": target_modules,
                "Result Dir": str(d),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        method_order = {
            "Whisper-base": 0,
            "Full Fine-tuning": 1,
            "CLAP-guided Full Fine-tuning": 2,
            "Ours": 3,
            "LoRA-Whisper +out_proj": 4,
            "LoRA-Whisper -out_proj": 5,
            "KAUST-style Residual Adapter": 6,
        }
        df["_order"] = df["Method"].map(method_order).fillna(99)
        df = df.sort_values(["_order", "LR", "Beam"], na_position="first").drop(columns="_order")

    print("=" * 140)
    if df.empty:
        print("No completed beam1/beam5 test summaries found under", root)
    else:
        print(df.to_markdown(index=False))
    print("=" * 140)

    csv_path = Path(args.csv) if args.csv else root / "common_asr_results.csv"
    md_path = Path(args.md) if args.md else root / "common_asr_results.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        if df.empty:
            f.write("No completed results found.\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")
    print("saved csv:", csv_path)
    print("saved md :", md_path)


if __name__ == "__main__":
    main()
