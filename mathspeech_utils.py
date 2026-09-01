import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


SOURCE_COLUMN_CANDIDATES = [
    "source",
    "Source",
    "source_id",
    "Source ID",
    "source_name",
    "Source Name",
    "speaker",
    "Speaker",
    "lecture",
    "Lecture",
    "video_id",
    "video",
    "url",
    "URL",
    "file",
    "filename",
    "audio",
    "audio_path",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_audio_16k(audio_path: str) -> np.ndarray:
    """Load audio as mono 16 kHz float32 using ffmpeg.

    This avoids mixing OpenAI Whisper's Python package into the baseline/eval stack.
    The returned array can be passed directly to Hugging Face WhisperProcessor.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]

    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def infer_source_ids(df, source_col: Optional[str] = None) -> Tuple[list[str], str]:
    """Return source ids used for source-aware splitting.

    If source_col is omitted, the function searches common source-like column names.
    It intentionally fails when no source column can be inferred, because falling back to
    sample-level splitting would reintroduce source leakage.
    """
    if source_col is not None:
        if source_col not in df.columns:
            raise ValueError(
                f"source_col={source_col!r} not found. Available columns: {df.columns.tolist()}"
            )
        return df[source_col].astype(str).tolist(), source_col

    for col in SOURCE_COLUMN_CANDIDATES:
        if col in df.columns:
            return df[col].astype(str).tolist(), col

    raise ValueError(
        "Cannot infer source column for source-aware split. "
        f"Available columns: {df.columns.tolist()}. "
        "Pass --source_col <column_name> explicitly."
    )


def make_or_load_source_aware_split(
    df,
    save_dir: str,
    split_path: Optional[str] = None,
    source_col: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    force_new: bool = False,
):
    """Create or load a deterministic source-aware train/valid/test split.

    The same source id never appears in more than one split. The split is saved as a
    torch .pt file containing both indices and source lists, making later eval scripts
    use exactly the same partition.
    """
    os.makedirs(save_dir, exist_ok=True)
    if split_path is None:
        split_path = os.path.join(save_dir, "split_indices.pt")

    if os.path.exists(split_path) and not force_new:
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        if split.get("split_type") == "source_aware":
            print("loaded source-aware split from:", split_path)
            return split
        print("[WARN] existing split is not source-aware; regenerating:", split_path)

    source_ids, used_source_col = infer_source_ids(df, source_col=source_col)

    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, source in enumerate(source_ids):
        source_to_indices[str(source)].append(idx)

    sources = sorted(source_to_indices.keys())
    random.Random(seed).shuffle(sources)

    n_sources = len(sources)
    n_train = int(n_sources * train_ratio)
    n_valid = int(n_sources * valid_ratio)

    train_sources = set(sources[:n_train])
    valid_sources = set(sources[n_train : n_train + n_valid])
    test_sources = set(sources[n_train + n_valid :])

    train_idx, valid_idx, test_idx = [], [], []
    for source, indices in source_to_indices.items():
        if source in train_sources:
            train_idx.extend(indices)
        elif source in valid_sources:
            valid_idx.extend(indices)
        else:
            test_idx.extend(indices)

    assert train_sources.isdisjoint(valid_sources)
    assert train_sources.isdisjoint(test_sources)
    assert valid_sources.isdisjoint(test_sources)

    split = {
        "split_type": "source_aware",
        "seed": seed,
        "source_col": used_source_col,
        "train_idx": sorted(train_idx),
        "valid_idx": sorted(valid_idx),
        "test_idx": sorted(test_idx),
        "train_sources": sorted(train_sources),
        "valid_sources": sorted(valid_sources),
        "test_sources": sorted(test_sources),
        "num_sources": n_sources,
        "num_samples": len(df),
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
    }

    torch.save(split, split_path)

    print("saved source-aware split to:", split_path)
    print("source_col:", used_source_col)
    print(
        f"sources: train={len(train_sources)}, valid={len(valid_sources)}, test={len(test_sources)}"
    )
    print(f"samples: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

    return split


def audio_path_for_index(audio_dir: str, real_idx: int, ext: str = "mp3") -> str:
    return str(Path(audio_dir) / f"{real_idx + 1}.{ext}")
