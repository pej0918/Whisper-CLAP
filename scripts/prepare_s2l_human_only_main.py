#!/usr/bin/env python
"""Prepare the final Human-only S2L ASR manifests without overwriting legacy Mix data.

Final filters:
    language == 'eng'
    is_tts == 0
    duration <= 30 sec

S2L-Sentences:
    keep the released official train/test boundary, then make a group-disjoint
    validation split from the official training set.

S2L-Equations:
    the released HF config does not expose the paper's exact indices, so make a
    deterministic formula-disjoint train/valid/test split with seed 42.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_s2l_asr_from_arrow import (
    add_global_sample_ids,
    assert_group_disjoint,
    attach_sample_ids,
    load_subset,
    split_equations_paper_style,
    split_train_val_group_disjoint,
    summarize,
)


def keep_human(df):
    out = df[df["is_tts"].astype(int) == 0].copy()
    if len(out) and (out["language"].astype(str).str.lower() != "eng").any():
        raise RuntimeError("Non-English rows survived Human-only filtering")
    if len(out) and (out["duration"].astype(float) > 30.0).any():
        raise RuntimeError("Rows longer than 30 sec survived filtering")
    return out


def save_split(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("sample_id").reset_index(drop=True).to_csv(path, index=False)
    print(f"saved {path}: {summarize(df)}")


def save_human_task(task_dir, prefix, all_df, train, valid, test, split_note):
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    train = attach_sample_ids(train, all_df)
    valid = attach_sample_ids(valid, all_df)
    test = attach_sample_ids(test, all_df)

    save_split(all_df, task_dir / f"{prefix}_all.csv")
    save_split(train, task_dir / f"{prefix}_train.csv")
    save_split(valid, task_dir / f"{prefix}_valid.csv")
    save_split(test, task_dir / f"{prefix}_test.csv")

    stats = {
        "main_protocol": "Human-only English ASR",
        "filters": {"language": "eng", "is_tts": 0, "max_duration_sec": 30.0},
        "split_note": split_note,
        "all": summarize(all_df),
        "train": summarize(train),
        "valid": summarize(valid),
        "test": summarize(test),
    }
    (task_dir / f"{prefix}_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def prepare_sent(args):
    task_dir = Path(args.out_root) / "s2l_sent"
    audio_root = task_dir / "audio"
    official_train = load_subset(
        Path(args.data_root) / "s2l_sentences_train", "sent", "official_train",
        audio_root, "eng", 30.0, args.allow_incomplete, args.extract_audio,
    )
    official_test = load_subset(
        Path(args.data_root) / "s2l_sentences_test", "sent", "official_test",
        audio_root, "eng", 30.0, args.allow_incomplete, args.extract_audio,
    )
    official_train = keep_human(official_train)
    official_test = keep_human(official_test)

    overlap = set(official_train.group_id) & set(official_test.group_id)
    print("Human-only official train/test sentence-group overlap:", len(overlap))
    if overlap:
        raise RuntimeError("S2L-Sentences official train/test group overlap after Human-only filtering")

    train, valid = split_train_val_group_disjoint(official_train, args.sent_val_ratio, args.seed)
    test = official_test
    assert_group_disjoint(train, valid, test, "S2L-Sent Human-only")

    all_df = add_global_sample_ids(pd.concat([official_train, official_test], ignore_index=True))
    save_human_task(
        task_dir, "s2l_sent", all_df, train, valid, test,
        "Official S2L-Sentences train/test boundary retained after English Human-only filtering; "
        f"official training sentence groups split train/valid with val_ratio={args.sent_val_ratio}, seed={args.seed}."
    )


def prepare_eq(args):
    task_dir = Path(args.out_root) / "s2l_eq"
    audio_root = task_dir / "audio"
    full = load_subset(
        Path(args.data_root) / "s2l_equations", "eq", "all",
        audio_root, "eng", 30.0, args.allow_incomplete, args.extract_audio,
    )
    full = keep_human(full)
    train, valid, test = split_equations_paper_style(
        full, val_ratio=args.eq_val_ratio, test_ratio=args.eq_test_ratio, seed=args.seed
    )
    assert_group_disjoint(train, valid, test, "S2L-Eq Human-only")
    all_df = add_global_sample_ids(full)
    save_human_task(
        task_dir, "s2l_eq", all_df, train, valid, test,
        "Human-only English S2L-Equations; deterministic formula-disjoint reconstructed split "
        f"with test_ratio={args.eq_test_ratio}, total val_ratio={args.eq_val_ratio}, seed={args.seed}."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data1/eunju/datasets/speech2latex_data")
    ap.add_argument("--out_root", default="/data1/eunju/datasets/speech2latex_asr_human_seed42")
    ap.add_argument("--task", choices=["sent", "eq", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sent_val_ratio", type=float, default=0.10)
    ap.add_argument("--eq_val_ratio", type=float, default=0.10)
    ap.add_argument("--eq_test_ratio", type=float, default=0.10)
    ap.add_argument("--extract_audio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow_incomplete", action="store_true")
    args = ap.parse_args()

    print("=" * 88)
    print("FINAL S2L HUMAN-ONLY PREPARATION")
    print("filters: language=eng, is_tts=0, duration<=30 sec")
    print("out_root:", args.out_root)
    print("legacy Mix outputs are not modified")
    print("=" * 88)

    if args.task in {"sent", "all"}:
        prepare_sent(args)
    if args.task in {"eq", "all"}:
        prepare_eq(args)


if __name__ == "__main__":
    main()
