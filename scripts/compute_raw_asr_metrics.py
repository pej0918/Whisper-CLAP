import argparse
import pandas as pd
from jiwer import wer, cer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref_col", default="reference")
    ap.add_argument("--pred_col", default="prediction")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.ref_col not in df.columns:
        raise ValueError(
            f"Missing reference column: {args.ref_col}\n"
            f"Available: {list(df.columns)}"
        )

    if args.pred_col not in df.columns:
        raise ValueError(
            f"Missing prediction column: {args.pred_col}\n"
            f"Available: {list(df.columns)}"
        )

    refs = (
        df[args.ref_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    hyps = (
        df[args.pred_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Ignore empty references, same philosophy as current evaluator.
    valid = [
        (r, h)
        for r, h in zip(refs, hyps)
        if r
    ]

    refs = [x[0] for x in valid]
    hyps = [x[1] for x in valid]

    raw_wer = wer(refs, hyps)
    raw_cer = cer(refs, hyps)

    print("=" * 50)
    print("RAW ASR METRICS (NO TEXT NORMALIZATION)")
    print("=" * 50)
    print("CSV          :", args.csv)
    print("Samples      :", len(refs))
    print(f"Raw WER      : {raw_wer:.6f} ({100*raw_wer:.2f}%)")
    print(f"Raw CER      : {raw_cer:.6f} ({100*raw_cer:.2f}%)")


if __name__ == "__main__":
    main()
