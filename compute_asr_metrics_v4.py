import os
import re
import argparse
import pandas as pd


# =========================================================
# Text normalization
# =========================================================
NUM_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def separate_alnum(text):
    """
    5y -> 5 y
    x2 -> x 2
    3xy -> 3 xy
    """
    text = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", text)
    return text


def normalize_text(s):
    s = str(s).lower()
    s = separate_alnum(s)

    # punctuation 제거
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # 숫자 canonicalization
    tokens = s.split()
    norm_tokens = []
    for tok in tokens:
        if tok in NUM_MAP:
            norm_tokens.append(NUM_MAP[tok])
        else:
            norm_tokens.append(tok)

    s = " ".join(norm_tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================================================
# Edit distance / WER / CER
# =========================================================
def edit_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def wer(ref, hyp):
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(ref, hyp):
    ref_chars = list(normalize_text(ref).replace(" ", ""))
    hyp_chars = list(normalize_text(hyp).replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# =========================================================
# Math term metrics
# =========================================================
MATH_TERMS = [
    # calculus / algebra
    "limit", "integral", "derivative", "prime", "differential",
    "matrix", "determinant", "inverse", "adjoint",
    "eigenvalue", "eigenvector", "jacobian", "lagrange",

    # operations / expressions
    "plus", "minus", "times", "divided", "over", "equals",
    "squared", "square", "cubed", "cube", "root", "power",
    "sum", "sigma", "product",

    # trig / log / exp
    "sine", "cosine", "tangent", "tan", "log", "natural",
    "exponential",

    # symbols
    "alpha", "beta", "gamma", "delta", "theta", "lambda",
    "omega", "phi", "pi", "infinity",

    # differentials
    "dx", "dy", "dt",

    # common spoken numbers
    "zero", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten",
]


def get_terms(text):
    text = normalize_text(text)
    terms = set()

    for term in MATH_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text):
            terms.add(term)

    return terms


def math_term_recall(ref, hyp):
    ref_terms = get_terms(ref)
    hyp_terms = get_terms(hyp)

    if len(ref_terms) == 0:
        return None

    return len(ref_terms & hyp_terms) / len(ref_terms)


def over_bias_rate(ref, hyp):
    ref_terms = get_terms(ref)
    hyp_terms = get_terms(hyp)

    if len(hyp_terms) == 0:
        return 0.0

    false_terms = hyp_terms - ref_terms
    return len(false_terms) / len(hyp_terms)


# =========================================================
# Tail hallucination rate
# =========================================================
def has_tail_hallucination(hyp):
    h = str(hyp).lower()

    tail_markers = [
        " nd", " ndi", " ndx", " ndy", " ndt", " ndp",
        " more than", " of r", " of l", " of y", " of pi",
        " going to be", " the same as",
    ]

    if any(m in h for m in tail_markers):
        return 1

    if any(ord(ch) > 127 for ch in h):
        return 1

    words = normalize_text(h).split()

    # repeated unigram / phrase
    for n in [1, 2, 3]:
        if len(words) < n * 4:
            continue

        for i in range(len(words) - n * 3):
            p1 = words[i:i+n]
            p2 = words[i+n:i+2*n]
            p3 = words[i+2*n:i+3*n]

            if p1 == p2 == p3:
                return 1

    return 0


def length_ratio(ref, hyp):
    ref_len = len(normalize_text(ref).split())
    hyp_len = len(normalize_text(hyp).split())

    if ref_len == 0:
        return 0.0

    return hyp_len / ref_len


# =========================================================
# Utilities
# =========================================================
def find_pred_col(df):
    pred_cols = [c for c in df.columns if c.startswith("pred_")]

    if len(pred_cols) == 0:
        raise ValueError(f"No prediction column found. Columns: {df.columns.tolist()}")

    if len(pred_cols) > 1:
        print("[warning] multiple pred columns found:", pred_cols)
        print("[warning] using last one:", pred_cols[-1])

    return pred_cols[-1]


def compute_metrics(csv_path, pred_col=None, ref_col="transcription"):
    df = pd.read_csv(csv_path)

    if pred_col is None:
        pred_col = find_pred_col(df)

    eval_df = df[
        df[pred_col].notna()
        & (df[pred_col].astype(str).str.strip().str.len() > 0)
    ].copy()

    if len(eval_df) == 0:
        raise ValueError(f"No evaluated rows found in {csv_path} with pred_col={pred_col}")

    wers = []
    cers = []
    recalls = []
    over_biases = []
    tails = []
    ratios = []

    for _, row in eval_df.iterrows():
        ref = row[ref_col]
        hyp = row[pred_col]

        wers.append(wer(ref, hyp))
        cers.append(cer(ref, hyp))

        r = math_term_recall(ref, hyp)
        if r is not None:
            recalls.append(r)

        over_biases.append(over_bias_rate(ref, hyp))
        tails.append(has_tail_hallucination(hyp))
        ratios.append(length_ratio(ref, hyp))

    return {
        "csv": csv_path,
        "pred_col": pred_col,
        "num_eval": len(eval_df),
        "WER": sum(wers) / len(wers),
        "CER": sum(cers) / len(cers),
        "MathTermRecall": sum(recalls) / len(recalls) if len(recalls) > 0 else None,
        "OverBiasRate": sum(over_biases) / len(over_biases),
        "TailRate": sum(tails) / len(tails),
        "AvgLenRatio": sum(ratios) / len(ratios),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--pred_col", type=str, default=None)
    parser.add_argument("--ref_col", type=str, default="transcription")
    parser.add_argument("--out_csv", type=str, default=None)

    args = parser.parse_args()

    result = compute_metrics(
        csv_path=args.csv,
        pred_col=args.pred_col,
        ref_col=args.ref_col,
    )

    print("====================================")
    for k, v in result.items():
        print(f"{k}: {v}")
    print("====================================")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        pd.DataFrame([result]).to_csv(args.out_csv, index=False)
        print("saved metric to:", args.out_csv)


if __name__ == "__main__":
    main()
