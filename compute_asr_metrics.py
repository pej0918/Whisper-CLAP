import argparse
import os
import re

import pandas as pd
from jiwer import cer as jiwer_cer
from jiwer import process_words


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

MATH_TERMS = [
    "limit", "integral", "derivative", "prime", "differential",
    "matrix", "determinant", "inverse", "adjoint",
    "eigenvalue", "eigenvector", "jacobian", "lagrange",
    "plus", "minus", "times", "divided", "over", "equals",
    "squared", "square", "cubed", "cube", "root", "power",
    "sum", "sigma", "product",
    "sine", "cosine", "tangent", "tan", "log", "natural", "exponential",
    "alpha", "beta", "gamma", "delta", "theta", "lambda", "omega", "phi", "pi", "infinity",
    "dx", "dy", "dt",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
]

TAIL_ARTIFACT_TOKENS = {
    "ida", "ire", "ist", "ouch", "ader", "nu", "nd", "ndi", "ndx", "ndy", "ndt", "ndp",
}
TAIL_PHRASES = [
    " more than", " more ", " of r", " of l", " of y", " of pi",
    " going to be", " it is", " it's", " the same as", " right here", " okay",
]


def separate_alnum(text: str) -> str:
    text = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", text)
    return text


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = separate_alnum(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for tok in text.split():
        tokens.append(NUM_MAP.get(tok, tok))
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def clean_asr_tail(text: str) -> tuple[str, bool]:
    raw = str(text).strip().replace(" ,", ",").replace(" .", ".")
    raw = " ".join(raw.split())
    if raw == "":
        return "", False

    words = raw.split()
    if len(words) <= 3:
        return raw.strip(" ,."), False

    for i, word in enumerate(words):
        token = word.lower().strip(".,!?;:;\"'")
        if i >= 3 and any(ord(ch) > 127 for ch in word):
            return " ".join(words[:i]).strip(" ,."), True
        if i >= 3 and (token in TAIL_ARTIFACT_TOKENS or token.startswith(("nd", "ndi", "ndx", "ndy", "ndt", "ndp"))):
            return " ".join(words[:i]).strip(" ,."), True

    lowered = " " + raw.lower()
    best_cut = None
    for phrase in TAIL_PHRASES:
        pos = lowered.find(phrase)
        if pos != -1:
            prefix = raw[: max(pos - 1, 0)].strip()
            if len(prefix.split()) >= 3:
                best_cut = pos if best_cut is None else min(best_cut, pos)
    if best_cut is not None:
        return raw[: best_cut - 1].strip(" ,."), True

    for n in [6, 5, 4, 3, 2, 1]:
        if len(words) < n * 4:
            continue
        for i in range(len(words) - 3 * n + 1):
            p1 = [x.lower().strip(".,!?;:") for x in words[i : i + n]]
            p2 = [x.lower().strip(".,!?;:") for x in words[i + n : i + 2 * n]]
            p3 = [x.lower().strip(".,!?;:") for x in words[i + 2 * n : i + 3 * n]]
            if p1 == p2 == p3:
                return " ".join(words[: i + n]).strip(" ,."), True

    return raw.strip(" ,."), False


def get_terms(text: str) -> set[str]:
    text = normalize_text(text)
    return {term for term in MATH_TERMS if re.search(rf"\b{re.escape(term)}\b", text)}


def math_term_recall(ref: str, hyp: str):
    ref_terms = get_terms(ref)
    if len(ref_terms) == 0:
        return None
    hyp_terms = get_terms(hyp)
    return len(ref_terms & hyp_terms) / len(ref_terms)


def over_bias_rate(ref: str, hyp: str) -> float:
    ref_terms = get_terms(ref)
    hyp_terms = get_terms(hyp)
    if len(hyp_terms) == 0:
        return 0.0
    return len(hyp_terms - ref_terms) / len(hyp_terms)


def has_tail_hallucination(hyp: str) -> int:
    _, removed = clean_asr_tail(hyp)
    return int(removed)


def length_ratio(ref: str, hyp: str) -> float:
    ref_len = len(normalize_text(ref).split())
    hyp_len = len(normalize_text(hyp).split())
    if ref_len == 0:
        return 0.0
    return hyp_len / ref_len


def find_pred_col(df) -> str:
    pred_cols = [c for c in df.columns if c.startswith("pred_") or c.endswith("_pred")]
    if len(pred_cols) == 0:
        raise ValueError(f"No prediction column found. Columns: {df.columns.tolist()}")
    if len(pred_cols) > 1:
        print("[warning] multiple prediction columns found:", pred_cols)
        print("[warning] using last one:", pred_cols[-1])
    return pred_cols[-1]


def compute_metrics(csv_path, pred_col=None, ref_col="transcription", clean_tail=False):
    df = pd.read_csv(csv_path)
    if pred_col is None:
        pred_col = find_pred_col(df)

    eval_df = df[df[pred_col].notna()].copy()
    eval_df[pred_col] = eval_df[pred_col].astype(str)
    eval_df = eval_df[eval_df[pred_col].str.strip().str.len() > 0].copy()
    if len(eval_df) == 0:
        raise ValueError(f"No evaluated rows found in {csv_path} with pred_col={pred_col}")

    refs, hyps = [], []
    recalls, over_biases, tails, ratios = [], [], [], []

    for _, row in eval_df.iterrows():
        ref_raw = row[ref_col]
        hyp_raw = row[pred_col]
        hyp_cleaned, tail_removed = clean_asr_tail(hyp_raw) if clean_tail else (str(hyp_raw), False)

        ref = normalize_text(ref_raw)
        hyp = normalize_text(hyp_cleaned)
        refs.append(ref)
        hyps.append(hyp)

        recall = math_term_recall(ref_raw, hyp_cleaned)
        if recall is not None:
            recalls.append(recall)
        over_biases.append(over_bias_rate(ref_raw, hyp_cleaned))
        tails.append(int(tail_removed) if clean_tail else has_tail_hallucination(hyp_raw))
        ratios.append(length_ratio(ref_raw, hyp_cleaned))

    word_out = process_words(refs, hyps)
    char_refs = [r.replace(" ", "") for r in refs]
    char_hyps = [h.replace(" ", "") for h in hyps]

    total_ref_words = word_out.hits + word_out.substitutions + word_out.deletions
    return {
        "csv": csv_path,
        "pred_col": pred_col,
        "ref_col": ref_col,
        "num_eval": len(eval_df),
        "metric_level": "corpus",
        "WER": word_out.wer,
        "CER": jiwer_cer(char_refs, char_hyps),
        "MathTermRecall": sum(recalls) / len(recalls) if recalls else None,
        "OverBiasRate": word_out.insertions / max(total_ref_words, 1),
        "TailRate": sum(tails) / len(tails),
        "AvgLenRatio": sum(ratios) / len(ratios),
        "total_hits": word_out.hits,
        "total_substitutions": word_out.substitutions,
        "total_insertions": word_out.insertions,
        "total_deletions": word_out.deletions,
        "total_ref_words": total_ref_words,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--pred_col", type=str, default=None)
    parser.add_argument("--ref_col", type=str, default="transcription")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--clean_tail", action="store_true")
    args = parser.parse_args()

    result = compute_metrics(
        csv_path=args.csv,
        pred_col=args.pred_col,
        ref_col=args.ref_col,
        clean_tail=args.clean_tail,
    )

    print("====================================")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("====================================")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        pd.DataFrame([result]).to_csv(args.out_csv, index=False)
        print("saved metric to:", args.out_csv)


if __name__ == "__main__":
    main()
