#!/usr/bin/env python3
import argparse
import json
import random
import re
import subprocess
import tarfile
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm


def run(cmd, check=True):
    print("[CMD]", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check)


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("¢", "")
    text = text.replace("©", "")
    text = text.replace("Na*", "Na+")
    text = text.replace("K*", "K+")
    text = text.replace("Kt", "K+")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_spoken_csv(path: Path) -> str:
    """
    slide_000_spoken.csv:
        ,Word,Start,End
        605,you,1.79,1.99
        ...
    """
    if not path.exists():
        return ""

    try:
        df = pd.read_csv(path)
    except Exception:
        return ""

    if "Word" not in df.columns:
        return ""

    df = df.dropna(subset=["Word"])

    if "Start" in df.columns:
        df = df.sort_values("Start")

    words = df["Word"].astype(str).tolist()
    text = " ".join(words)
    return clean_text(text)


def read_ocr_csv(path: Path, conf_min: float = 0.0) -> str:
    """
    slide_000_ocr.csv:
        level,page_num,...,conf,text
    """
    if not path.exists():
        return ""

    try:
        df = pd.read_csv(path)
    except Exception:
        return ""

    if "text" not in df.columns:
        return ""

    df = df.dropna(subset=["text"])

    if "conf" in df.columns:
        df = df[df["conf"].fillna(-1).astype(float) >= conf_min]

    sort_cols = [
        c for c in ["block_num", "par_num", "line_num", "word_num"]
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)

    if all(c in df.columns for c in ["block_num", "par_num", "line_num"]):
        lines = []
        for _, g in df.groupby(["block_num", "par_num", "line_num"], sort=False):
            line = " ".join(g["text"].astype(str).tolist())
            line = clean_text(line)
            if line:
                lines.append(line)
        text = " ".join(lines)
    else:
        text = " ".join(df["text"].astype(str).tolist())

    return clean_text(text)


def make_caption(spoken: str, ocr: str, mode: str) -> str:
    spoken = clean_text(spoken)
    ocr = clean_text(ocr)

    if mode == "spoken":
        return f'The lecturer says: "{spoken}"' if spoken else ""

    if mode == "ocr":
        return f"Slide text: {ocr}" if ocr else ""

    if mode == "spoken_ocr":
        parts = []
        if spoken:
            parts.append(f'The lecturer says: "{spoken}"')
        if ocr:
            parts.append(f"Slide text: {ocr}")
        return " ".join(parts)

    raise ValueError(f"Unknown caption mode: {mode}")


def parse_time_token(tok: str) -> float:
    tok = tok.strip()
    if ":" not in tok:
        return float(tok)

    parts = [float(x) for x in tok.split(":")]
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss

    raise ValueError(f"Cannot parse time token: {tok}")


def get_spoken_duration(spoken_csv: Path) -> float | None:
    """
    slide_xxx_spoken.csv의 End 최대값을 slide duration으로 사용.
    spoken time은 slide segment 내부 relative time으로 보임.
    """
    if not spoken_csv.exists():
        return None

    try:
        df = pd.read_csv(spoken_csv)
    except Exception:
        return None

    if "End" not in df.columns:
        return None

    end_vals = pd.to_numeric(df["End"], errors="coerce").dropna()
    if len(end_vals) == 0:
        return None

    return float(end_vals.max())


def parse_segments(path: Path, lecture_dir: Path | None = None):
    """
    MLP segments.txt 포맷:
        25.4677
        100.0046
        150.0046
        ...

    해석:
        각 줄은 slide_i의 absolute start time.
        slide_i end time은 다음 줄의 start time.
        마지막 slide end time은 slide_i_spoken.csv의 max End를 더해서 추정.

    반환:
        {slide_num: (start_sec, end_sec)}
    """
    starts = []

    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            starts.append(float(line))
        except ValueError:
            nums = re.findall(r"\d+(?:\.\d+)?", line)
            if nums:
                starts.append(float(nums[0]))

    if len(starts) < 2:
        raise ValueError(f"Not enough segment timestamps: {path}")

    segment_map = {}

    for i in range(len(starts) - 1):
        start = starts[i]
        end = starts[i + 1]

        if end > start:
            segment_map[i] = (start, end)

    last_slide = len(starts) - 1
    last_start = starts[-1]

    last_end = None
    if lecture_dir is not None:
        spoken_csv = lecture_dir / f"slide_{last_slide:03d}_spoken.csv"
        duration = get_spoken_duration(spoken_csv)

        if duration is not None and duration > 0:
            last_end = last_start + duration

    if last_end is not None and last_end > last_start:
        segment_map[last_slide] = (last_start, last_end)
    else:
        print(
            f"[WARN] Could not infer end time for last slide {last_slide:03d}: {path}. "
            f"Last slide will be skipped."
        )

    return segment_map


def get_video_id_from_lecture_dir(lecture_dir: Path) -> str | None:
    files = list(lecture_dir.glob("*_transcripts.csv"))
    if not files:
        return None

    name = files[0].name
    return name.replace("_transcripts.csv", "")


def download_audio(video_id: str, full_audio_dir: Path) -> Path | None:
    full_audio_dir.mkdir(parents=True, exist_ok=True)

    final_wav = full_audio_dir / f"{video_id}.wav"
    if final_wav.exists() and final_wav.stat().st_size > 0:
        return final_wav

    url = f"https://www.youtube.com/watch?v={video_id}"
    tmp_template = str(full_audio_dir / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        tmp_template,
        url,
    ]

    try:
        run(cmd)
    except subprocess.CalledProcessError:
        print(f"[WARN] yt-dlp failed: {url}")
        return None

    if final_wav.exists():
        return final_wav

    candidates = list(full_audio_dir.glob(f"{video_id}.*"))
    for c in candidates:
        if c.suffix.lower() == ".wav":
            return c

    print(f"[WARN] downloaded audio not found for video_id={video_id}")
    return None


def cut_audio_to_flac(
    input_audio: Path,
    output_flac: Path,
    start: float,
    end: float,
    sample_rate: int = 48000,
):
    output_flac.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end - start)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        str(input_audio),
        "-t",
        str(duration),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(output_flac),
    ]
    run(cmd)


def is_valid_audio(path: Path, min_sec: float = 0.1) -> tuple[bool, str]:
    """
    torchaudio.load()로 실제 decode 가능한 audio인지 확인.
    TorchCodec/FFmpeg에서 frame decode가 안 되는 flac를 사전에 제거하기 위함.
    """
    if not path.exists():
        return False, "missing"

    if path.stat().st_size == 0:
        return False, "empty file"

    try:
        wav, sr = torchaudio.load(str(path))
    except Exception as e:
        return False, f"decode failed: {repr(e)}"

    if sr is None or sr <= 0:
        return False, f"invalid sample rate: {sr}"

    if wav.numel() == 0 or wav.shape[-1] <= 0:
        return False, "zero frames"

    duration = wav.shape[-1] / sr
    if duration < min_sec:
        return False, f"too short: {duration:.4f}s"

    return True, "ok"


def find_lecture_dirs(root: Path):
    return sorted([p.parent for p in root.rglob("segments.txt")])


def split_lecture_dirs(lecture_dirs, train_ratio=0.9, valid_ratio=0.05, seed=42):
    rng = random.Random(seed)
    lecture_dirs = list(lecture_dirs)
    rng.shuffle(lecture_dirs)

    n = len(lecture_dirs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    split_map = {}

    for p in lecture_dirs[:n_train]:
        split_map[p] = "train"

    for p in lecture_dirs[n_train:n_train + n_valid]:
        split_map[p] = "valid"

    for p in lecture_dirs[n_train + n_valid:]:
        split_map[p] = "test"

    return split_map


def sanitize_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def collect_slide_numbers(lecture_dir: Path):
    nums = []
    for p in lecture_dir.glob("slide_*_spoken.csv"):
        m = re.search(r"slide_(\d+)_spoken\.csv", p.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(set(nums))


def make_processed_dataset(args):
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    processed_dir = out / "processed"
    full_audio_dir = out / "full_audio"

    lecture_dirs = find_lecture_dirs(root)
    if args.max_lectures is not None:
        lecture_dirs = lecture_dirs[:args.max_lectures]

    print(f"[INFO] found lecture dirs: {len(lecture_dirs)}")

    split_map = split_lecture_dirs(
        lecture_dirs,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    manifest_path = out / "manifest.jsonl"
    out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    n_bad_audio = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for lecture_dir in tqdm(lecture_dirs, desc="lectures"):
            split = split_map[lecture_dir]
            video_id = get_video_id_from_lecture_dir(lecture_dir)

            if not video_id:
                print(f"[WARN] no transcript csv/video id: {lecture_dir}")
                n_skip += 1
                continue

            try:
                segment_map = parse_segments(lecture_dir / "segments.txt", lecture_dir=lecture_dir)
            except Exception as e:
                print(f"[WARN] segment parse failed: {lecture_dir} / {e}")
                n_skip += 1
                continue

            if not segment_map:
                print(f"[WARN] empty segment map: {lecture_dir}")
                n_skip += 1
                continue

            if args.dry_run:
                print(f"[DRY] {lecture_dir}")
                print(f"      video_id={video_id}")
                print(f"      segments example={list(segment_map.items())[:3]}")
                print(f"      slides example={collect_slide_numbers(lecture_dir)[:5]}")
                continue

            input_audio = None
            if args.download:
                input_audio = download_audio(video_id, full_audio_dir)
            else:
                candidate = full_audio_dir / f"{video_id}.wav"
                if candidate.exists():
                    input_audio = candidate

            if input_audio is None or not input_audio.exists():
                print(f"[WARN] missing audio for {video_id}. Use --download or put {video_id}.wav in {full_audio_dir}")
                n_skip += 1
                continue

            rel_parts = lecture_dir.relative_to(root).parts
            lecture_key_prefix = sanitize_key("_".join(rel_parts))

            slide_nums = collect_slide_numbers(lecture_dir)

            for slide_num in slide_nums:
                if slide_num not in segment_map:
                    print(f"[WARN] no segment for slide {slide_num}: {lecture_dir}")
                    n_skip += 1
                    continue

                spoken_path = lecture_dir / f"slide_{slide_num:03d}_spoken.csv"
                ocr_path = lecture_dir / f"slide_{slide_num:03d}_ocr.csv"

                spoken = read_spoken_csv(spoken_path)
                ocr = read_ocr_csv(ocr_path, conf_min=args.ocr_conf_min)
                caption = make_caption(spoken, ocr, args.caption_mode)

                if len(caption.split()) < args.min_words:
                    n_skip += 1
                    continue

                start, end = segment_map[slide_num]
                if end - start < args.min_duration:
                    n_skip += 1
                    continue

                sample_key = sanitize_key(f"{lecture_key_prefix}_slide_{slide_num:03d}")

                split_dir = processed_dir / split
                flac_path = split_dir / f"{sample_key}.flac"
                json_path = split_dir / f"{sample_key}.json"

                if not flac_path.exists():
                    try:
                        cut_audio_to_flac(
                            input_audio=input_audio,
                            output_flac=flac_path,
                            start=start,
                            end=end,
                            sample_rate=args.sample_rate,
                        )
                    except subprocess.CalledProcessError:
                        print(f"[WARN] ffmpeg failed: {sample_key}")
                        n_skip += 1
                        continue

                ok_audio, audio_msg = is_valid_audio(flac_path, min_sec=args.min_audio_sec)
                if not ok_audio:
                    print(f"[WARN] invalid audio skipped: {sample_key} / {audio_msg}")
                    flac_path.unlink(missing_ok=True)
                    json_path.unlink(missing_ok=True)
                    n_skip += 1
                    n_bad_audio += 1
                    continue

                metadata = {
                    "text": [caption],
                    "tag": ["lecture", "slide", "spoken_language"],
                    "original_data": {
                        "dataset": "MLP",
                        "lecture_dir": str(lecture_dir),
                        "video_id": video_id,
                        "slide_number": slide_num,
                        "start": start,
                        "end": end,
                        "spoken_csv": str(spoken_path),
                        "ocr_csv": str(ocr_path),
                        "caption_mode": args.caption_mode,
                    },
                }

                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                record = {
                    "key": sample_key,
                    "split": split,
                    "audio": str(flac_path),
                    "json": str(json_path),
                    "text": caption,
                    "video_id": video_id,
                    "slide_number": slide_num,
                    "start": start,
                    "end": end,
                    "lecture_dir": str(lecture_dir),
                }
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += 1

    print(f"[DONE] processed samples: {n_ok}")
    print(f"[DONE] skipped: {n_skip}")
    print(f"[DONE] bad audio skipped: {n_bad_audio}")
    print(f"[DONE] manifest: {manifest_path}")
    print(f"[DONE] processed dir: {processed_dir}")

    return processed_dir


def make_tars(processed_dir: Path, tar_out_dir: Path, num_element: int = 512):
    processed_dir = Path(processed_dir)
    tar_out_dir = Path(tar_out_dir)
    tar_out_dir.mkdir(parents=True, exist_ok=True)

    for split_dir in sorted(processed_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        split = split_dir.name
        json_files = sorted(split_dir.glob("*.json"))
        split_tar_dir = tar_out_dir / split
        split_tar_dir.mkdir(parents=True, exist_ok=True)

        sizes = {}
        chunk_idx = 0
        n_excluded = 0

        for i in range(0, len(json_files), num_element):
            chunk = json_files[i:i + num_element]
            tar_name = f"mlp_{split}_{chunk_idx:06d}.tar"
            tar_path = split_tar_dir / tar_name

            with tarfile.open(tar_path, "w") as tar:
                count = 0
                for json_path in chunk:
                    key = json_path.stem
                    flac_path = json_path.with_suffix(".flac")

                    if not flac_path.exists():
                        n_excluded += 1
                        continue

                    ok_audio, audio_msg = is_valid_audio(flac_path)
                    if not ok_audio:
                        print(f"[WARN] invalid audio excluded from tar: {flac_path} / {audio_msg}")
                        n_excluded += 1
                        continue

                    tar.add(flac_path, arcname=f"{key}.flac")
                    tar.add(json_path, arcname=f"{key}.json")
                    count += 1

            sizes[tar_name] = count
            chunk_idx += 1

        with (split_tar_dir / "sizes.json").open("w", encoding="utf-8") as f:
            json.dump(sizes, f, indent=2)

        print(
            f"[TAR] {split}: {len(json_files)} json files -> {split_tar_dir} "
            f"/ excluded={n_excluded}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="/data1/eunju/datasets/data_oct",
        help="MLP dataset root",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/data1/eunju/datasets/mlp_clap",
        help="output directory",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download full YouTube audio with yt-dlp",
    )
    parser.add_argument(
        "--caption_mode",
        type=str,
        default="spoken_ocr",
        choices=["spoken", "ocr", "spoken_ocr"],
    )
    parser.add_argument("--ocr_conf_min", type=float, default=0.0)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--min_words", type=int, default=3)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--min_audio_sec", type=float, default=0.1)

    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--make_tars", action="store_true")
    parser.add_argument("--num_element", type=int, default=512)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_lectures", type=int, default=None)

    args = parser.parse_args()

    processed_dir = make_processed_dataset(args)

    if args.make_tars and not args.dry_run:
        tar_out_dir = Path(args.out).resolve() / "webdataset"
        make_tars(
            processed_dir=processed_dir,
            tar_out_dir=tar_out_dir,
            num_element=args.num_element,
        )


if __name__ == "__main__":
    main()