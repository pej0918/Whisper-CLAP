#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def read_best_valid_wer(run_dir: Path):
    summary = run_dir / "training_summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Missing: {summary}")
    data = json.loads(summary.read_text())
    for key in ("best_valid_wer", "best_dev_wer"):
        if data.get(key) is not None:
            return float(data[key]), data
    raise KeyError(f"No best validation WER in {summary}")


def main():
    ap = argparse.ArgumentParser(
        description="Select a learning-rate run strictly by best validation WER."
    )
    ap.add_argument("--root", required=True, help="Root containing METHOD_lrLR directories")
    ap.add_argument("--method", required=True)
    ap.add_argument("--lrs", nargs="+", default=["1e-5", "1e-4", "3e-4"])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for lr in args.lrs:
        run_dir = root / f"{args.method}_lr{lr}"
        wer, summary = read_best_valid_wer(run_dir)
        rows.append({
            "method": args.method,
            "lr": lr,
            "best_valid_wer": wer,
            "best_epoch": summary.get("best_epoch"),
            "run_dir": str(run_dir.resolve()),
        })

    rows.sort(key=lambda x: x["best_valid_wer"])
    selected = rows[0]
    result = {
        "selection_metric": "validation WER",
        "candidate_lrs": args.lrs,
        "runs": rows,
        "selected_lr": selected["lr"],
        "selected_best_valid_wer": selected["best_valid_wer"],
        "selected_run_dir": selected["run_dir"],
    }

    output = Path(args.output) if args.output else root / f"{args.method}_lr_selection.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"METHOD: {args.method}")
    for row in rows:
        mark = "  <-- SELECTED" if row["lr"] == selected["lr"] else ""
        print(f"LR={row['lr']:<7} best_valid_wer={row['best_valid_wer']:.6f}{mark}")
    print(f"selected_lr={selected['lr']}")
    print(f"selected_run_dir={selected['run_dir']}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
