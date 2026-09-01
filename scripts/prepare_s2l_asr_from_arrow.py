import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset
from sklearn.model_selection import train_test_split


FORMULA_TAG_RE = re.compile(r"</?formula_(?:start|end)>", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def clean_pronunciation(text):
    """Speech2LaTeX pronunciation -> spoken-math ASR target."""
    text = "" if text is None else str(text)
    text = FORMULA_TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def canonical_text(text):
    text = "" if text is None else str(text)
    return SPACE_RE.sub(" ", text).strip()


def make_group_id(task, latex_reference):
    """Group every recording/voice of the same equation or sentence together."""
    digest = hashlib.sha1(canonical_text(latex_reference).encode("utf-8")).hexdigest()[:20]
    return f"{task}_{digest}"


def find_arrow_files(path, allow_incomplete=False):
    path = Path(path)
    files = sorted(path.glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow files found: {path}")

    pat = re.compile(r"data-(\d+)-of-(\d+)\.arrow$")
    indices = set()
    expected = None
    for f in files:
        m = pat.match(f.name)
        if not m:
            continue
        indices.add(int(m.group(1)))
        total = int(m.group(2))
        if expected is None:
            expected = total
        elif expected != total:
            raise RuntimeError(f"Inconsistent shard count in {path}: {expected} vs {total}")

    if expected is not None:
        missing = sorted(set(range(expected)) - indices)
        if missing and not allow_incomplete:
            raise RuntimeError(
                f"Incomplete dataset in {path}: found {len(indices)}/{expected} shards, missing={missing}. "
                "Wait until the Hugging Face download finishes, or use --allow_incomplete for a smoke test only."
            )
        if missing:
            print(f"WARNING: partial dataset in {path}; missing shards={missing}")
    return files


def get_audio_info(blob):
    with sf.SoundFile(io.BytesIO(blob), "r") as f:
        sr = int(f.samplerate)
        frames = int(len(f))
    return sr, frames, frames / sr


def save_audio(blob, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.write_bytes(blob)


def load_subset(
    input_dir,
    task,
    subset_name,
    audio_root,
    language="eng",
    max_duration=30.0,
    allow_incomplete=False,
    extract_audio=True,
):
    files = find_arrow_files(input_dir, allow_incomplete=allow_incomplete)
    rows = []
    stats = Counter()

    required = {
        "spk", "language", "sentence_id", "is_tts", "pronunciation",
        "whisper_text", "sentence", "audio_path",
    }

    for shard_idx, arrow_path in enumerate(files):
        print("\n" + "=" * 100)
        print(f"[{task}/{subset_name}] {shard_idx + 1}/{len(files)} {arrow_path.name}")

        ds = Dataset.from_file(str(arrow_path))
        missing = required - set(ds.column_names)
        if missing:
            raise RuntimeError(f"Missing columns in {arrow_path}: {sorted(missing)}")

        # Avoid librosa decoding; access embedded bytes directly.
        ds = ds.cast_column("audio_path", Audio(decode=False))

        for local_idx in range(len(ds)):
            r = ds[local_idx]
            stats["seen"] += 1

            if str(r["language"]).lower() != language.lower():
                stats["skip_language"] += 1
                continue

            reference_text = clean_pronunciation(r["pronunciation"])
            if not reference_text:
                stats["skip_empty_text"] += 1
                continue

            audio_obj = r["audio_path"]
            if not isinstance(audio_obj, dict) or audio_obj.get("bytes") is None:
                raise RuntimeError(f"No embedded audio bytes at {arrow_path}:{local_idx}")

            blob = audio_obj["bytes"]
            sr, frames, duration = get_audio_info(blob)
            if duration <= 0:
                stats["skip_zero_audio"] += 1
                continue
            if max_duration is not None and duration > max_duration:
                stats["skip_long"] += 1
                continue

            original_audio_name = str(audio_obj.get("path") or "audio.wav")
            suffix = Path(original_audio_name).suffix or ".wav"
            audio_filename = f"{arrow_path.stem}__row{local_idx:06d}{suffix}"
            output_audio = Path(audio_root) / subset_name / audio_filename
            if extract_audio:
                save_audio(blob, output_audio)

            latex_reference = str(r["sentence"])
            group_id = make_group_id(task, latex_reference)

            rows.append({
                "group_id": group_id,
                "video_id": group_id,
                "source": group_id,
                "audio_path": str(output_audio.resolve()),
                "start": 0.0,
                "end": float(duration),
                "duration": float(duration),
                "reference_text": reference_text,
                "text_spoken": reference_text,
                "pronunciation_raw": str(r["pronunciation"]),
                "whisper_text_original": str(r["whisper_text"]),
                "latex_reference": latex_reference,
                "spk": str(r["spk"]),
                "language": str(r["language"]),
                "sentence_id": int(r["sentence_id"]),
                "is_tts": int(r["is_tts"]),
                "audio_sr": sr,
                "audio_frames": frames,
                "formula_info": str(r.get("formula_info", "")),
                "count_formula_symbols": r.get("count_formula_symbols", None),
                "source_subset": subset_name,
                "source_shard": arrow_path.name,
                "source_row": local_idx,
                "original_audio_name": original_audio_name,
            })
            stats["kept"] += 1

    df = pd.DataFrame(rows)
    print(f"\n[{task}/{subset_name}] rows={len(df)} groups={df.group_id.nunique() if len(df) else 0}")
    if len(df):
        print("annotation:", dict(Counter("H" if x == 0 else "A" for x in df.is_tts)))
        print(f"hours={df.duration.sum()/3600:.2f}")
    print("scan stats:", dict(stats))
    return df


def split_train_val_group_disjoint(df, val_ratio, seed):
    groups = sorted(df.group_id.unique().tolist())
    if len(groups) < 2:
        raise RuntimeError("Need >=2 groups for train/validation split")

    train_groups, val_groups = train_test_split(
        groups,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )
    train_groups, val_groups = set(train_groups), set(val_groups)
    return (
        df[df.group_id.isin(train_groups)].copy(),
        df[df.group_id.isin(val_groups)].copy(),
    )


def split_equations_paper_style(df, val_ratio, test_ratio, seed):
    """
    Paper-style formula-disjoint split.

    The released HF S2L-Equations config does not expose the paper's exact sample indices.
    We therefore follow the public Speech2LaTeX/MathBridge split convention:
      unique formulas -> dev/test with sklearn train_test_split(..., random_state=42),
      then dev formulas -> train/validation.

    All human/TTS recordings of one equation remain in the same split.
    """
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Require val_ratio > 0, test_ratio > 0, and val_ratio + test_ratio < 1")

    groups = sorted(df.group_id.unique().tolist())
    if len(groups) < 3:
        raise RuntimeError("Need >=3 unique equation groups")

    dev_groups, test_groups = train_test_split(
        groups,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )

    # val_ratio is a fraction of the total dataset, so rescale inside dev.
    val_ratio_in_dev = val_ratio / (1.0 - test_ratio)
    train_groups, val_groups = train_test_split(
        dev_groups,
        test_size=val_ratio_in_dev,
        random_state=seed,
        shuffle=True,
    )

    train_groups, val_groups, test_groups = map(set, (train_groups, val_groups, test_groups))
    return (
        df[df.group_id.isin(train_groups)].copy(),
        df[df.group_id.isin(val_groups)].copy(),
        df[df.group_id.isin(test_groups)].copy(),
    )


def assert_group_disjoint(train, valid, test, name):
    tr, va, te = set(train.group_id), set(valid.group_id), set(test.group_id)
    overlap = {
        "train-valid": len(tr & va),
        "train-test": len(tr & te),
        "valid-test": len(va & te),
    }
    print(f"[{name}] group overlap: {overlap}")
    if any(overlap.values()):
        raise RuntimeError(f"{name}: group leakage detected: {overlap}")


def add_global_sample_ids(df):
    df = df.reset_index(drop=True).copy()
    df.insert(0, "sample_id", range(1, len(df) + 1))
    return df


def attach_sample_ids(split_df, all_df):
    keys = ["source_subset", "source_shard", "source_row"]
    ids = all_df[keys + ["sample_id"]]
    out = split_df.merge(ids, on=keys, how="left", validate="one_to_one")
    if out.sample_id.isna().any():
        raise RuntimeError("Failed to assign global sample IDs")
    out["sample_id"] = out.sample_id.astype(int)
    return out


def annotation_subset(df, annotation):
    if annotation == "mix":
        return df.copy()
    if annotation == "h":
        return df[df.is_tts == 0].copy()
    if annotation == "a":
        return df[df.is_tts == 1].copy()
    raise ValueError(annotation)


def summarize(df):
    if len(df) == 0:
        return {"rows": 0, "groups": 0, "hours": 0.0, "human": 0, "artificial": 0}
    return {
        "rows": int(len(df)),
        "groups": int(df.group_id.nunique()),
        "hours": float(df.duration.sum() / 3600),
        "human": int((df.is_tts == 0).sum()),
        "artificial": int((df.is_tts == 1).sum()),
        "max_duration": float(df.duration.max()),
    }


def save_manifest(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("sample_id").reset_index(drop=True).to_csv(path, index=False)
    print(f"saved {path.name}: {summarize(df)}")


def save_all_variants(task_dir, prefix, all_df, train, valid, test, split_note):
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    train = attach_sample_ids(train, all_df)
    valid = attach_sample_ids(valid, all_df)
    test = attach_sample_ids(test, all_df)

    save_manifest(all_df, task_dir / f"{prefix}_all.csv")
    save_manifest(train, task_dir / f"{prefix}_train_mix.csv")
    save_manifest(valid, task_dir / f"{prefix}_valid_mix.csv")
    save_manifest(test, task_dir / f"{prefix}_test_mix.csv")
    save_manifest(annotation_subset(test, "h"), task_dir / f"{prefix}_test_h.csv")
    save_manifest(annotation_subset(test, "a"), task_dir / f"{prefix}_test_a.csv")
    save_manifest(annotation_subset(train, "h"), task_dir / f"{prefix}_train_h.csv")
    save_manifest(annotation_subset(valid, "h"), task_dir / f"{prefix}_valid_h.csv")
    save_manifest(annotation_subset(train, "a"), task_dir / f"{prefix}_train_a.csv")
    save_manifest(annotation_subset(valid, "a"), task_dir / f"{prefix}_valid_a.csv")

    stats = {
        "split_note": split_note,
        "all": summarize(all_df),
        "train_mix": summarize(train),
        "valid_mix": summarize(valid),
        "test_mix": summarize(test),
        "test_h": summarize(annotation_subset(test, "h")),
        "test_a": summarize(annotation_subset(test, "a")),
        "train_h": summarize(annotation_subset(train, "h")),
        "train_a": summarize(annotation_subset(train, "a")),
    }
    (task_dir / f"{prefix}_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def prepare_sentences(args):
    print("\n" + "#" * 100)
    print("PREPARE S2L-SENTENCES")
    print("#" * 100)

    task_dir = Path(args.out_root) / "s2l_sent"
    audio_root = task_dir / "audio"

    official_train = load_subset(
        Path(args.data_root) / "s2l_sentences_train",
        "sent", "official_train", audio_root,
        args.language, args.max_duration, args.allow_incomplete, args.extract_audio,
    )
    official_test = load_subset(
        Path(args.data_root) / "s2l_sentences_test",
        "sent", "official_test", audio_root,
        args.language, args.max_duration, args.allow_incomplete, args.extract_audio,
    )

    overlap = set(official_train.group_id) & set(official_test.group_id)
    print("Official train/test sentence overlap:", len(overlap))
    if overlap:
        raise RuntimeError("Official S2L-Sent train/test are not sentence-disjoint after filtering")

    train, valid = split_train_val_group_disjoint(official_train, args.sent_val_ratio, args.seed)
    test = official_test
    assert_group_disjoint(train, valid, test, "S2L-Sent")

    all_df = add_global_sample_ids(pd.concat([official_train, official_test], ignore_index=True))
    save_all_variants(
        task_dir,
        "s2l_sent",
        all_df,
        train,
        valid,
        test,
        split_note=(
            "Official S2L-Sentences train/test partition retained; official training sentences are "
            f"split group-disjoint into train/validation with val_ratio={args.sent_val_ratio}, seed={args.seed}."
        ),
    )


def prepare_equations(args):
    print("\n" + "#" * 100)
    print("PREPARE S2L-EQUATIONS")
    print("#" * 100)

    task_dir = Path(args.out_root) / "s2l_eq"
    audio_root = task_dir / "audio"

    full = load_subset(
        Path(args.data_root) / "s2l_equations",
        "eq", "all", audio_root,
        args.language, args.max_duration, args.allow_incomplete, args.extract_audio,
    )

    train, valid, test = split_equations_paper_style(
        full,
        val_ratio=args.eq_val_ratio,
        test_ratio=args.eq_test_ratio,
        seed=args.seed,
    )
    assert_group_disjoint(train, valid, test, "S2L-Eq")

    all_df = add_global_sample_ids(full)
    save_all_variants(
        task_dir,
        "s2l_eq",
        all_df,
        train,
        valid,
        test,
        split_note=(
            "Paper-style formula-disjoint split because the released HF S2L-Equations config does not expose "
            "the paper's exact sample indices. Unique equations are split with sklearn train_test_split: "
            f"test_ratio={args.eq_test_ratio}, total val_ratio={args.eq_val_ratio}, seed={args.seed}."
        ),
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Prepare Speech2LaTeX HF Arrow shards for spoken-mathematics ASR. "
            "English audio is used as input; the provided pronunciation is the ASR target."
        )
    )
    ap.add_argument("--data_root", default="/data1/eunju/datasets/speech2latex_data")
    ap.add_argument("--out_root", default="/data1/eunju/datasets/speech2latex_asr_seed42")
    ap.add_argument("--task", choices=["sent", "eq", "all"], default="all")
    ap.add_argument("--language", default="eng")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sent_val_ratio", type=float, default=0.10)
    ap.add_argument("--eq_val_ratio", type=float, default=0.10)
    ap.add_argument("--eq_test_ratio", type=float, default=0.10)
    ap.add_argument("--max_duration", type=float, default=30.0)
    ap.add_argument("--extract_audio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Use currently downloaded shards only. Smoke-test use only.",
    )
    args = ap.parse_args()

    if args.max_duration <= 0:
        args.max_duration = None

    print("=" * 100)
    print("Speech2LaTeX -> spoken-math ASR")
    print("=" * 100)
    print("data_root    :", args.data_root)
    print("out_root     :", args.out_root)
    print("language     :", args.language)
    print("seed         :", args.seed)
    print("ASR GT       : pronunciation")
    print("formula tags : <formula_start>/<formula_end> removed")
    print("H/A/Mix      : preserved as separate manifests")
    print("split unit   : unique equation/sentence")

    if args.task in {"sent", "all"}:
        prepare_sentences(args)
    if args.task in {"eq", "all"}:
        prepare_equations(args)


if __name__ == "__main__":
    main()
