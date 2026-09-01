import argparse
from pathlib import Path

import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "sample_id",
    "video_id",
    "audio_path",
    "start",
    "end",
    "duration",
    "text_spoken",
]


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in [".jsonl", ".json"]:
        return pd.read_json(path, lines=(path.suffix.lower() == ".jsonl"))
    if path.suffix.lower() in [".tsv"]:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def prepare_one(input_path, output_path, args, split_name):
    df = read_table(input_path)

    def col(name, fallback=None):
        if name and name in df.columns:
            return df[name]
        if fallback is not None:
            return fallback
        raise ValueError(f"Column not found in {input_path}: {name}\nAvailable columns: {df.columns.tolist()}")

    audio = col(args.audio_col)
    if args.audio_root:
        audio = audio.map(lambda x: str(Path(args.audio_root) / str(x)) if not str(x).startswith("/") else str(x))

    if args.start_col and args.end_col:
        start = col(args.start_col).astype(float)
        end = col(args.end_col).astype(float)
        duration = end - start
    elif args.duration_col:
        start = pd.Series([0.0] * len(df))
        duration = col(args.duration_col).astype(float)
        end = duration
    else:
        start = pd.Series([0.0] * len(df))
        duration = pd.Series([args.default_duration] * len(df))
        end = duration

    sample_id = col(args.sample_id_col, pd.Series([f"{split_name}_{i}" for i in range(len(df))])).astype(str)
    group_source = args.group_col if args.group_col in df.columns else None
    video_id = col(group_source, sample_id).astype(str)
    text = col(args.text_col).astype(str)

    out = pd.DataFrame({
        "sample_id": sample_id,
        "video_id": video_id,
        "audio_path": audio.astype(str),
        "start": start.astype(float),
        "end": end.astype(float),
        "duration": duration.astype(float),
        "text_spoken": text,
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"saved {split_name}: {output_path} ({len(out)} samples)")
    print("columns:", REQUIRED_OUTPUT_COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="Convert original Speech2LaTeX split files into the team Whisper-CLAP manifest format."
    )
    parser.add_argument("--train_in", required=True)
    parser.add_argument("--valid_in", required=True)
    parser.add_argument("--test_in", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--audio_root", default=None)
    parser.add_argument("--audio_col", default="audio_path")
    parser.add_argument("--text_col", default="text_spoken")
    parser.add_argument("--sample_id_col", default="sample_id")
    parser.add_argument("--group_col", default="video_id", help="Use the original split/group key when available; otherwise sample_id is used.")
    parser.add_argument("--start_col", default=None)
    parser.add_argument("--end_col", default=None)
    parser.add_argument("--duration_col", default="duration")
    parser.add_argument("--default_duration", type=float, default=30.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    prepare_one(args.train_in, out_dir / "s2l_train.csv", args, "train")
    prepare_one(args.valid_in, out_dir / "s2l_valid.csv", args, "valid")
    prepare_one(args.test_in, out_dir / "s2l_test.csv", args, "test")

    train = pd.read_csv(out_dir / "s2l_train.csv")
    valid = pd.read_csv(out_dir / "s2l_valid.csv")
    test = pd.read_csv(out_dir / "s2l_test.csv")
    tr, va, te = set(train["video_id"].astype(str)), set(valid["video_id"].astype(str)), set(test["video_id"].astype(str))
    print("group overlap")
    print("train-valid:", len(tr & va))
    print("train-test :", len(tr & te))
    print("valid-test :", len(va & te))
    print("NOTE: if the original Speech2LaTeX split intentionally shares group ids, this check is only diagnostic.")


if __name__ == "__main__":
    main()
