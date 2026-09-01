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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    root = Path(args.output_dir)
    methods = [
        ("Whisper-base", root / "whisper_base", "no training; beam=5 eval"),
        ("Full Fine-tuning", root / "fullft", "Seq2SeqTrainer; CE only; lr=1e-5; epoch=10"),
        ("Ours", root / "ours", "team projector/adapter pipeline"),
        ("LoRA-Whisper fair", root / "lora_fair_lr1e-5", "r=32; q/k/v/fc1/fc2; lr=1e-5; epoch=10"),
        ("LoRA-Whisper paper", root / "lora_paper_lr1e-4", "r=32; q/k/v/fc1/fc2; lr=1e-4; epoch=10"),
        ("KAUST-style Residual Adapter", root / "residual_b256", "Whisper frozen; layer-wise residual adapter; b=256; lr=1e-5; epoch=10"),
    ]

    rows = []
    for name, d, setting in methods:
        test = load_json(d / "test_summary_beam5.json")
        train = load_json(d / "training_summary.json")
        status = "DONE" if test else "PENDING"
        rows.append({
            "Method": name,
            "Status": status,
            "Trainable Params": None if not train else train.get("trainable_params"),
            "WER": None if not test else test.get("wer"),
            "CER": None if not test else test.get("cer"),
            "Best Valid WER": None if not train else train.get("best_dev_wer", train.get("best_valid_wer")),
            "Num Eval": None if not test else test.get("samples", test.get("valid_samples")),
            "RTF": None if not test else test.get("rtf"),
            "Setting": setting,
            "Result Dir": str(d),
        })

    df = pd.DataFrame(rows)
    print("=" * 120)
    print(df.to_markdown(index=False))
    print("=" * 120)

    csv_path = Path(args.csv) if args.csv else root / "common_asr_results.csv"
    md_path = Path(args.md) if args.md else root / "common_asr_results.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print("saved csv:", csv_path)
    print("saved md :", md_path)


if __name__ == "__main__":
    main()
