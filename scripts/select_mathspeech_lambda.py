import argparse
import json
from pathlib import Path

import pandas as pd


def load_candidate(run_dir: Path):
    summary_path = run_dir / "training_summary.json"
    log_path = run_dir / "train_log.csv"
    ckpt_path = run_dir / "best.pt"
    if not (summary_path.is_file() and log_path.is_file() and ckpt_path.is_file()):
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    args = summary.get("args", {})
    lam_a = float(args.get("lambda_align", 0.0))
    lam_h = float(args.get("lambda_hidden", 0.0))
    best_wer = float(summary["best_valid_wer"])
    best_epoch = int(summary["best_epoch"])

    log = pd.read_csv(log_path)
    row = log.loc[log["epoch"].astype(int) == best_epoch]
    valid_ce = float(row.iloc[0]["valid_ce"]) if len(row) and "valid_ce" in row.columns else float("inf")

    return {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "lambda_align": lam_a,
        "lambda_hidden": lam_h,
        "best_epoch": best_epoch,
        "best_valid_wer": best_wer,
        "valid_ce_at_best_wer": valid_ce,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search_root", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_env", required=True)
    ap.add_argument("--ranking_csv", required=True)
    ap.add_argument("--extra_dir", action="append", default=[])
    args = ap.parse_args()

    root = Path(args.search_root)
    candidates = []

    for p in sorted(root.iterdir()):
        if p.is_dir():
            item = load_candidate(p)
            if item is not None:
                candidates.append(item)

    for x in args.extra_dir:
        p = Path(x)
        if p.exists():
            item = load_candidate(p)
            if item is not None:
                candidates.append(item)

    # De-duplicate exact lambda pairs, preferring search-root candidates.
    dedup = {}
    for c in candidates:
        key = (c["lambda_align"], c["lambda_hidden"])
        if key not in dedup:
            dedup[key] = c
    candidates = list(dedup.values())

    if not candidates:
        raise RuntimeError(f"No complete candidate runs found under {root}")

    # Primary selection: validation WER.
    # Tie-break 1: validation CE at the checkpoint selected by validation WER.
    # Tie-break 2: smaller total auxiliary-loss weight.
    candidates.sort(
        key=lambda x: (
            x["best_valid_wer"],
            x["valid_ce_at_best_wer"],
            x["lambda_align"] + x["lambda_hidden"],
            x["lambda_hidden"],
            x["lambda_align"],
        )
    )

    for rank, c in enumerate(candidates, start=1):
        c["rank"] = rank

    best = candidates[0]

    ranking_path = Path(args.ranking_csv)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(ranking_path, index=False)

    output = {
        "selection_metric": "minimum validation WER at beam=5",
        "tie_break": "validation CE at best-WER epoch, then smaller auxiliary weight",
        "selected": best,
        "num_candidates": len(candidates),
        "ranking_csv": str(ranking_path.resolve()),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    env_path = Path(args.output_env)
    with open(env_path, "w") as f:
        f.write(f"SELECTED_ALIGN={best['lambda_align']}\n")
        f.write(f"SELECTED_HIDDEN={best['lambda_hidden']}\n")
        f.write(f"SELECTED_CKPT='{best['checkpoint']}'\n")
        f.write(f"SELECTED_DIR='{best['run_dir']}'\n")
        f.write(f"SELECTED_VALID_WER={best['best_valid_wer']}\n")
        f.write(f"SELECTED_BEST_EPOCH={best['best_epoch']}\n")

    print("=" * 88)
    print("MATHSPEECH LAMBDA SEARCH — VALIDATION RANKING")
    print("=" * 88)
    print(f"{'Rank':>4} {'lambda_a':>10} {'lambda_h':>10} {'Epoch':>7} {'Valid WER':>12} {'Valid CE':>12}")
    for c in candidates:
        print(
            f"{c['rank']:>4d} {c['lambda_align']:>10.4g} {c['lambda_hidden']:>10.4g} "
            f"{c['best_epoch']:>7d} {c['best_valid_wer']:>12.6f} {c['valid_ce_at_best_wer']:>12.6f}"
        )
    print("-" * 88)
    print("SELECTED")
    print(f"lambda_align  = {best['lambda_align']}")
    print(f"lambda_hidden = {best['lambda_hidden']}")
    print(f"best_epoch    = {best['best_epoch']}")
    print(f"valid_WER     = {best['best_valid_wer']:.6f}")
    print(f"checkpoint    = {best['checkpoint']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
