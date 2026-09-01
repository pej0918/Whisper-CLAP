import argparse
import os
import re
import string
from collections import Counter

import pandas as pd


# -----------------------------
# 1. Prediction cleaning
# -----------------------------

TAIL_ARTIFACT_TOKENS = {
    "ida", "ire", "ist", "ouch", "ader",
    "nu", "nd", "ndi", "ndx", "ndy", "ndt", "ndp",
}

TAIL_PHRASES = [
    " more than",
    " more ",
    " of r",
    " of l",
    " of y",
    " of pi",
    " going to be",
    " it is",
    " it's",
    " the same as",
    " right here",
    " okay",
    " so this is",
    " that's a big",
]


def basic_space_clean(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = text.replace(" ,", ",").replace(" .", ".")
    text = " ".join(text.split())
    return text.strip()


def clean_asr_tail(text: str):
    """
    Returns:
        cleaned_text: str
        tail_removed: bool

    tail_removed=True only when hallucinated/repeated tail is cut.
    Simple whitespace/punctuation cleanup alone does not count as tail removal.
    """
    raw = basic_space_clean(text)
    if raw == "":
        return "", False

    text = raw
    words = text.split()
    tail_removed = False

    if len(words) <= 3:
        return text.strip(" ,."), False

    # 1) non-English / broken unicode tail 제거
    for i, w in enumerate(words):
        if i >= 3 and any(ord(ch) > 127 for ch in w):
            return " ".join(words[:i]).strip(" ,."), True

    # 2) ida / ire / ouch / ist / ndx 류 artifact tail 제거
    for i, w in enumerate(words):
        ww = w.lower().strip(".,!?;:;\"'")
        if i >= 3:
            if ww in TAIL_ARTIFACT_TOKENS:
                return " ".join(words[:i]).strip(" ,."), True
            if any(ww.startswith(p) for p in ["nd", "ndi", "ndx", "ndy", "ndt", "ndp"]):
                return " ".join(words[:i]).strip(" ,."), True

    # 3) 자주 붙는 hallucination phrase 제거
    lowered = " " + text.lower()
    best_cut = None

    for phrase in TAIL_PHRASES:
        pos = lowered.find(phrase)
        if pos != -1:
            prefix = text[:max(pos - 1, 0)].strip()
            if len(prefix.split()) >= 3:
                if best_cut is None or pos < best_cut:
                    best_cut = pos

    if best_cut is not None:
        return text[:best_cut - 1].strip(" ,."), True

    # 4) repeated phrase / repeated n-gram tail 제거
    # ex) "y equals 3t y equals 3t y equals 3t"
    words = text.split()

    for n in range(8, 0, -1):
        if len(words) < n * 3:
            continue

        norm_words = [w.lower().strip(".,!?;:;\"'") for w in words]

        for i in range(len(words) - 3 * n + 1):
            p1 = norm_words[i:i + n]
            p2 = norm_words[i + n:i + 2 * n]
            p3 = norm_words[i + 2 * n:i + 3 * n]

            if p1 == p2 == p3:
                # 첫 phrase 하나만 남김
                return " ".join(words[:i + n]).strip(" ,."), True

    # 5) 마지막 단어가 과도하게 반복되는 경우
    # ex) "prime prime prime prime"
    norm_words = [w.lower().strip(".,!?;:;\"'") for w in words]
    if len(norm_words) >= 6:
        last = norm_words[-1]
        repeat_count = 0
        for w in reversed(norm_words):
            if w == last:
                repeat_count += 1
            else:
                break

        if repeat_count >= 4:
            cut = len(words) - repeat_count + 1
            return " ".join(words[:cut]).strip(" ,."), True

    return text.strip(" ,."), tail_removed


# -----------------------------
# 2. Normalization
# -----------------------------

def normalize_for_words(text: str):
    text = basic_space_clean(text).lower()

    # common unicode normalization
    text = text.replace("−", " minus ")
    text = text.replace("-", " minus ")

    # remove punctuation as word separators
    punct = string.punctuation
    for p in punct:
        text = text.replace(p, " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str):
    norm = normalize_for_words(text)
    if norm == "":
        return []
    return norm.split()


def normalize_for_chars(text: str):
    text = normalize_for_words(text)
    # CER는 공백 제거 후 계산
    return text.replace(" ", "")


# -----------------------------
# 3. Edit distance
# -----------------------------

def levenshtein_distance(ref, hyp):
    """
    ref, hyp: list or string
    returns edit distance only
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution / correct
            )

    return dp[n][m]


def word_error_counts(ref_words, hyp_words):
    """
    returns:
        distance, substitutions, insertions, deletions
    """
    n, m = len(ref_words), len(hyp_words)

    # dp[i][j] = (cost, S, I, D)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)

    for i in range(1, n + 1):
        c, s, ins, d = dp[i - 1][0]
        dp[i][0] = (c + 1, s, ins, d + 1)

    for j in range(1, m + 1):
        c, s, ins, d = dp[0][j - 1]
        dp[0][j] = (c + 1, s, ins + 1, d)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = []

            # deletion
            c, s, ins, d = dp[i - 1][j]
            candidates.append((c + 1, s, ins, d + 1))

            # insertion
            c, s, ins, d = dp[i][j - 1]
            candidates.append((c + 1, s, ins + 1, d))

            # correct or substitution
            c, s, ins, d = dp[i - 1][j - 1]
            if ref_words[i - 1] == hyp_words[j - 1]:
                candidates.append((c, s, ins, d))
            else:
                candidates.append((c + 1, s + 1, ins, d))

            # tie-break: lower cost, then fewer insertions, then fewer deletions
            dp[i][j] = min(candidates, key=lambda x: (x[0], x[2], x[3], x[1]))

    return dp[n][m]


# -----------------------------
# 4. Math term recall
# -----------------------------

MATH_TERMS = [
    # variables / symbols spoken as words
    "x", "y", "z", "a", "b", "c", "n", "m", "t", "r", "u", "v", "w",

    # constants / greek
    "pi", "theta", "phi", "lambda", "omega", "delta", "rho",

    # operators
    "plus", "minus", "times", "divided", "over", "equals", "equal",
    "greater", "less", "zero",

    # powers / roots
    "squared", "square", "cubed", "cube", "power", "root",
    "sqrt",

    # calculus
    "integral", "derivative", "prime", "partial", "gradient",
    "limit", "sum", "infinity",

    # functions
    "sine", "sin", "cosine", "cos", "tangent", "tan",
    "secant", "sec", "arctan", "arcsine",
    "log", "ln", "natural", "exponential", "exp",

    # algebra / vector
    "matrix", "determinant", "inverse", "adjoint",
    "dot", "cross", "vector",

    # fractions / numbers often important in math speech
    "half", "third", "fourth", "fifth",
]


ALIASES = {
    "cos": "cosine",
    "sin": "sine",
    "tan": "tangent",
    "sec": "secant",
    "ln": "log",
    "sqrt": "root",
    "equal": "equals",
    "exponential": "exp",
}


def canonical_math_token(tok: str):
    tok = tok.lower().strip()
    return ALIASES.get(tok, tok)


def extract_math_terms(text: str):
    words = tokenize_words(text)
    terms = []
    math_set = set(MATH_TERMS)

    for w in words:
        cw = canonical_math_token(w)
        if cw in math_set:
            terms.append(cw)

    return terms


def math_term_recall(ref_text: str, hyp_text: str):
    ref_terms = extract_math_terms(ref_text)
    hyp_terms = extract_math_terms(hyp_text)

    if len(ref_terms) == 0:
        return None, 0, 0

    ref_counter = Counter(ref_terms)
    hyp_counter = Counter(hyp_terms)

    matched = 0
    for term, cnt in ref_counter.items():
        matched += min(cnt, hyp_counter.get(term, 0))

    return matched / len(ref_terms), matched, len(ref_terms)


# -----------------------------
# 5. Full metric computation
# -----------------------------

def compute_metrics_for_column(df, pred_col, gt_col="transcription"):
    total_word_err = 0
    total_ref_words = 0

    total_char_err = 0
    total_ref_chars = 0

    total_insertions = 0

    total_math_hit = 0
    total_math_ref = 0

    tail_count = 0
    len_ratios = []

    cleaned_preds = []

    for _, row in df.iterrows():
        gt = str(row[gt_col]) if not pd.isna(row[gt_col]) else ""
        raw_pred = str(row[pred_col]) if not pd.isna(row[pred_col]) else ""

        pred, tail_removed = clean_asr_tail(raw_pred)
        cleaned_preds.append(pred)

        if tail_removed:
            tail_count += 1

        ref_words = tokenize_words(gt)
        hyp_words = tokenize_words(pred)

        dist, sub, ins, dele = word_error_counts(ref_words, hyp_words)

        total_word_err += dist
        total_insertions += ins
        total_ref_words += len(ref_words)

        ref_chars = normalize_for_chars(gt)
        hyp_chars = normalize_for_chars(pred)

        total_char_err += levenshtein_distance(ref_chars, hyp_chars)
        total_ref_chars += len(ref_chars)

        recall, hit, ref_total = math_term_recall(gt, pred)
        total_math_hit += hit
        total_math_ref += ref_total

        if len(ref_words) > 0:
            len_ratios.append(len(hyp_words) / len(ref_words))

    wer = total_word_err / max(total_ref_words, 1)
    cer = total_char_err / max(total_ref_chars, 1)
    math_recall = total_math_hit / max(total_math_ref, 1)

    # OverBias = word-level insertion rate
    overbias = total_insertions / max(total_ref_words, 1)

    # TailRate = samples where hallucinated tail was removed
    tailrate = tail_count / max(len(df), 1)

    avg_len_ratio = sum(len_ratios) / max(len(len_ratios), 1)

    metrics = {
        "pred_col": pred_col,
        "num_samples": len(df),
        "WER": wer,
        "CER": cer,
        "MathTerm_Recall": math_recall,
        "OverBias": overbias,
        "TailRate": tailrate,
        "AvgLenRatio": avg_len_ratio,
        "total_word_errors": total_word_err,
        "total_ref_words": total_ref_words,
        "total_insertions": total_insertions,
        "total_char_errors": total_char_err,
        "total_ref_chars": total_ref_chars,
        "total_math_hit": total_math_hit,
        "total_math_ref": total_math_ref,
        "tail_count": tail_count,
    }

    return metrics, cleaned_preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--gt_col", type=str, default="transcription")
    parser.add_argument(
        "--pred_cols",
        type=str,
        nargs="*",
        default=None,
        help="Prediction columns. If omitted, all columns starting with pred_ are used.",
    )
    parser.add_argument("--save_prefix", type=str, default="asr_metrics")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.pred_cols is None or len(args.pred_cols) == 0:
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
    else:
        pred_cols = args.pred_cols

    print("Prediction columns:", pred_cols)

    all_metrics = []
    cleaned_df = df.copy()

    for pred_col in pred_cols:
        if pred_col not in df.columns:
            print(f"[skip] missing column: {pred_col}")
            continue

        metrics, cleaned_preds = compute_metrics_for_column(
            df=df,
            pred_col=pred_col,
            gt_col=args.gt_col,
        )

        all_metrics.append(metrics)
        cleaned_df[f"{pred_col}_clean"] = cleaned_preds

        print()
        print("=" * 60)
        print(pred_col)
        print("=" * 60)
        print(f"WER              : {metrics['WER']:.4f}")
        print(f"CER              : {metrics['CER']:.4f}")
        print(f"MathTerm Recall  : {metrics['MathTerm_Recall']:.4f}")
        print(f"OverBias         : {metrics['OverBias']:.4f}")
        print(f"TailRate         : {metrics['TailRate']:.4f}")
        print(f"AvgLenRatio      : {metrics['AvgLenRatio']:.4f}")
        print(f"tail_count       : {metrics['tail_count']} / {metrics['num_samples']}")

    metric_df = pd.DataFrame(all_metrics)

    out_dir = os.path.dirname(args.save_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    metric_path = f"{args.save_prefix}_summary.csv"
    cleaned_path = f"{args.save_prefix}_cleaned_predictions.csv"

    metric_df.to_csv(metric_path, index=False)
    cleaned_df.to_csv(cleaned_path, index=False)

    print()
    print("saved metrics :", metric_path)
    print("saved cleaned :", cleaned_path)


if __name__ == "__main__":
    main()