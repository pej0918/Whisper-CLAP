import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from mathspeech_utils import (
    audio_path_for_index,
    load_audio_16k,
    make_or_load_source_aware_split,
)


S2L_DATASET_NAME = "marsianin500/Speech2Latex"
LANGUAGE_ALIASES = {
    "en": {"en", "eng", "english"},
    "eng": {"en", "eng", "english"},
    "english": {"en", "eng", "english"},
    "ru": {"ru", "rus", "russian"},
    "rus": {"ru", "rus", "russian"},
    "russian": {"ru", "rus", "russian"},
}


@dataclass
class ASRRecord:
    uid: str
    text: str
    group: str
    audio: object = None
    audio_path: Optional[str] = None
    metadata: Optional[Dict] = None


class ASRDataset(Dataset):
    def __init__(self, records: Sequence[ASRRecord], processor, audio_ext: str = "mp3"):
        self.records = list(records)
        self.processor = processor
        self.audio_ext = audio_ext

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        audio = load_record_audio_16k(rec)
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(
            str(rec.text).strip(),
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "input_features": input_features,
            "labels": labels,
            "text": str(rec.text).strip(),
            "uid": rec.uid,
            "metadata": rec.metadata or {},
        }


class ASRCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = torch.stack([x["input_features"] for x in batch], dim=0)
        labels = pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        return {
            "input_features": input_features,
            "labels": labels,
            "texts": [x["text"] for x in batch],
            "uids": [x["uid"] for x in batch],
            "metadata": [x["metadata"] for x in batch],
        }


def _as_numpy_1d(audio) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        # Audio can be [time, channel] or [channel, time]. Average channels.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    return arr.reshape(-1).astype(np.float32)


def resample_to_16k(audio, sampling_rate: int) -> np.ndarray:
    audio = _as_numpy_1d(audio)
    if int(sampling_rate) == 16000:
        return audio

    try:
        import torchaudio.functional as AF

        wav = torch.tensor(audio, dtype=torch.float32)
        wav = AF.resample(wav, orig_freq=int(sampling_rate), new_freq=16000)
        return wav.cpu().numpy().astype(np.float32)
    except Exception:
        pass

    try:
        import librosa

        return librosa.resample(audio, orig_sr=int(sampling_rate), target_sr=16000).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Audio sampling_rate={sampling_rate}; install torchaudio or librosa for resampling."
        ) from exc


def load_record_audio_16k(record: ASRRecord) -> np.ndarray:
    if record.audio is not None:
        audio_obj = record.audio
        if isinstance(audio_obj, dict):
            if "array" in audio_obj and audio_obj["array"] is not None:
                sr = int(audio_obj.get("sampling_rate", 16000))
                return resample_to_16k(audio_obj["array"], sr)
            if "path" in audio_obj and audio_obj["path"]:
                return load_audio_16k(audio_obj["path"])
        return _as_numpy_1d(audio_obj)

    if record.audio_path:
        return load_audio_16k(record.audio_path)

    raise ValueError(f"Record has no audio or audio_path: {record.uid}")


def _normalize_lang(value: object) -> str:
    return str(value).strip().lower()


def _wanted_language(value: object, wanted: str) -> bool:
    wanted = _normalize_lang(wanted)
    aliases = LANGUAGE_ALIASES.get(wanted, {wanted})
    return _normalize_lang(value) in aliases


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    columns = list(columns)
    lower_map = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def make_group_split(
    records: Sequence[ASRRecord],
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    group_to_uids: Dict[str, List[str]] = {}
    for r in records:
        group_to_uids.setdefault(str(r.group), []).append(r.uid)

    groups = sorted(group_to_uids.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    n_train = min(max(n_train, 1), max(n - 2, 1)) if n >= 3 else max(n - 1, 1)
    n_valid = min(max(n_valid, 1), max(n - n_train - 1, 0)) if n >= 3 else 0

    train_groups = set(groups[:n_train])
    valid_groups = set(groups[n_train : n_train + n_valid])
    test_groups = set(groups[n_train + n_valid :])

    split = {"train": [], "valid": [], "test": []}
    for r in records:
        g = str(r.group)
        if g in train_groups:
            split["train"].append(r.uid)
        elif g in valid_groups:
            split["valid"].append(r.uid)
        elif g in test_groups:
            split["test"].append(r.uid)
    return split


def _records_by_uid(records: Sequence[ASRRecord]) -> Dict[str, ASRRecord]:
    return {r.uid: r for r in records}


def load_mathspeech_records(
    split: str,
    excel_path: str,
    audio_dir: str,
    split_path: Optional[str],
    save_dir: str,
    source_col: Optional[str] = "Source",
    text_col: str = "transcription",
    audio_ext: str = "mp3",
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    force_new_split: bool = False,
) -> List[ASRRecord]:
    df = pd.read_excel(excel_path)
    split_obj = make_or_load_source_aware_split(
        df=df,
        save_dir=save_dir,
        split_path=split_path,
        source_col=source_col,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        force_new=force_new_split,
    )
    key = {"validation": "valid", "val": "valid"}.get(split, split)
    idx_key = f"{key}_idx"
    if idx_key not in split_obj:
        raise ValueError(f"Unknown split={split}; expected train/valid/test")

    records = []
    for real_idx in split_obj[idx_key]:
        text = _safe_text(df[text_col].iloc[real_idx])
        records.append(
            ASRRecord(
                uid=f"mathspeech:{real_idx}",
                text=text,
                group=str(df[source_col].iloc[real_idx]) if source_col and source_col in df.columns else str(real_idx),
                audio_path=audio_path_for_index(audio_dir, real_idx, ext=audio_ext),
                metadata={
                    "dataset": "mathspeech",
                    "real_idx": int(real_idx),
                    "source": str(df[source_col].iloc[real_idx]) if source_col and source_col in df.columns else "",
                },
            )
        )
    return records


def _load_s2l_dataset_dict(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc
    return load_dataset(dataset_name)


def _pick_s2l_split_keys(keys: Sequence[str], subset: str) -> Tuple[List[str], List[str]]:
    subset = subset.lower()
    if subset in {"sent", "sentence", "sentences", "s2l-sent", "s2l-sentences"}:
        tags = ["sent"]
    elif subset in {"eq", "equation", "equations", "s2l-eq", "s2l-equations"}:
        tags = ["equation", "eq"]
    else:
        raise ValueError(f"Unknown s2l_subset={subset}")

    def matches(k: str) -> bool:
        kl = k.lower()
        return any(t in kl for t in tags)

    train_keys = [k for k in keys if matches(k) and "train" in k.lower()]
    test_keys = [k for k in keys if matches(k) and "test" in k.lower()]
    all_keys = [k for k in keys if matches(k)]

    if not train_keys:
        train_keys = all_keys
    return train_keys, test_keys


def _s2l_record_from_example(
    example: Dict,
    uid: str,
    target_col: str,
    group_col: Optional[str],
    audio_col: str,
    subset: str,
) -> Optional[ASRRecord]:
    text = _safe_text(example.get(target_col))
    if not text:
        return None

    columns = example.keys()
    if group_col is None:
        group_col = _first_existing_column(
            columns,
            [
                "sentence_id",
                "equation_id",
                "formula_id",
                "id",
                "sentence_normalized",
                "formula_normalized",
                "sentence",
                target_col,
            ],
        )
    group_value = example.get(group_col) if group_col else text

    audio_obj = example.get(audio_col)
    audio_path = None
    audio = None
    if isinstance(audio_obj, dict):
        audio = audio_obj
    elif isinstance(audio_obj, str):
        audio_path = audio_obj
    else:
        audio = audio_obj

    return ASRRecord(
        uid=uid,
        text=text,
        group=f"{subset}:{group_value}",
        audio=audio,
        audio_path=audio_path,
        metadata={
            "dataset": "speech2latex",
            "subset": subset,
            "target_col": target_col,
            "group_col": group_col or "",
            "group": str(group_value),
            "language": str(example.get("language", "")),
            "is_tts": int(example.get("is_tts", -1)) if str(example.get("is_tts", "")).strip() != "" else -1,
        },
    )


def _collect_s2l_records(
    dataset_dict,
    split_keys: Sequence[str],
    subset: str,
    target_col: str,
    language: str,
    human_only: bool,
    group_col: Optional[str],
    audio_col: str,
    max_samples: Optional[int] = None,
) -> List[ASRRecord]:
    records: List[ASRRecord] = []
    for split_key in split_keys:
        ds = dataset_dict[split_key]
        for i, ex in enumerate(ds):
            if "language" in ex and not _wanted_language(ex["language"], language):
                continue
            if human_only and "is_tts" in ex and int(ex["is_tts"]) != 0:
                continue
            rec = _s2l_record_from_example(
                example=ex,
                uid=f"s2l:{split_key}:{i}",
                target_col=target_col,
                group_col=group_col,
                audio_col=audio_col,
                subset=subset,
            )
            if rec is not None:
                records.append(rec)
                if max_samples is not None and len(records) >= max_samples:
                    return records
    return records


def load_s2l_records(
    split: str,
    save_dir: str,
    split_path: Optional[str] = None,
    dataset_name: str = S2L_DATASET_NAME,
    subset: str = "sentences",
    target_col: str = "pronunciation",
    language: str = "eng",
    human_only: bool = True,
    group_col: Optional[str] = None,
    audio_col: str = "audio_path",
    seed: int = 42,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.1,
    force_new_split: bool = False,
    max_train_samples: Optional[int] = None,
    max_valid_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> List[ASRRecord]:
    os.makedirs(save_dir, exist_ok=True)
    if split_path is None:
        split_path = os.path.join(save_dir, f"s2l_{subset}_eng_human_split.pt")

    dataset_dict = _load_s2l_dataset_dict(dataset_name)
    keys = list(dataset_dict.keys())
    train_keys, test_keys = _pick_s2l_split_keys(keys, subset)

    train_pool = _collect_s2l_records(
        dataset_dict=dataset_dict,
        split_keys=train_keys,
        subset=subset,
        target_col=target_col,
        language=language,
        human_only=human_only,
        group_col=group_col,
        audio_col=audio_col,
        max_samples=None,
    )
    test_pool = _collect_s2l_records(
        dataset_dict=dataset_dict,
        split_keys=test_keys,
        subset=subset,
        target_col=target_col,
        language=language,
        human_only=human_only,
        group_col=group_col,
        audio_col=audio_col,
        max_samples=None,
    ) if test_keys else []

    all_records = train_pool + test_pool
    uid_to_record = _records_by_uid(all_records)

    if os.path.exists(split_path) and not force_new_split:
        split_obj = torch.load(split_path, map_location="cpu", weights_only=False)
    else:
        if test_pool:
            # Preserve official test split when available. Split only the train pool into train/valid groups.
            test_groups = {r.group for r in test_pool}
            train_pool_no_leak = [r for r in train_pool if r.group not in test_groups]
            train_valid_split = make_group_split(
                train_pool_no_leak,
                seed=seed,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
            )
            split_obj = {
                "split_type": "s2l_group_aware_official_test",
                "seed": seed,
                "dataset_name": dataset_name,
                "subset": subset,
                "target_col": target_col,
                "language": language,
                "human_only": human_only,
                "train_keys": train_keys,
                "test_keys": test_keys,
                "train": train_valid_split["train"],
                "valid": train_valid_split["valid"],
                "test": [r.uid for r in test_pool],
            }
        else:
            split_obj = make_group_split(
                train_pool,
                seed=seed,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
            )
            split_obj.update(
                {
                    "split_type": "s2l_group_aware_no_official_test",
                    "seed": seed,
                    "dataset_name": dataset_name,
                    "subset": subset,
                    "target_col": target_col,
                    "language": language,
                    "human_only": human_only,
                    "train_keys": train_keys,
                    "test_keys": test_keys,
                }
            )
        torch.save(split_obj, split_path)
        print("saved S2L split to:", split_path)

    key = {"validation": "valid", "val": "valid"}.get(split, split)
    if key not in split_obj:
        raise ValueError(f"Unknown split={split}; expected train/valid/test")

    records = [uid_to_record[uid] for uid in split_obj[key] if uid in uid_to_record]
    if key == "train" and max_train_samples is not None:
        records = records[:max_train_samples]
    if key == "valid" and max_valid_samples is not None:
        records = records[:max_valid_samples]
    if key == "test" and max_test_samples is not None:
        records = records[:max_test_samples]

    print(
        f"S2L {subset} {key}: records={len(records)} "
        f"language={language} human_only={human_only} target={target_col}"
    )
    return records


def add_dataset_args(parser):
    parser.add_argument("--dataset_type", type=str, default="mathspeech", choices=["mathspeech", "s2l"])
    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--text_col", type=str, default="transcription")
    parser.add_argument("--audio_ext", type=str, default="mp3")
    parser.add_argument("--source_col", type=str, default="Source")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--force_new_split", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)

    parser.add_argument("--s2l_dataset_name", type=str, default=S2L_DATASET_NAME)
    parser.add_argument("--s2l_subset", type=str, default="sentences", choices=["sentences", "equations"])
    parser.add_argument("--s2l_target_col", type=str, default="pronunciation")
    parser.add_argument("--s2l_language", type=str, default="eng")
    parser.add_argument("--s2l_include_tts", action="store_true")
    parser.add_argument("--s2l_group_col", type=str, default=None)
    parser.add_argument("--s2l_audio_col", type=str, default="audio_path")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser


def load_records_from_args(args, split: str, save_dir: Optional[str] = None) -> List[ASRRecord]:
    save_dir = save_dir or getattr(args, "save_dir", ".")
    if args.dataset_type == "mathspeech":
        return load_mathspeech_records(
            split=split,
            excel_path=args.excel_path,
            audio_dir=args.audio_dir,
            split_path=args.split_path,
            save_dir=save_dir,
            source_col=args.source_col,
            text_col=args.text_col,
            audio_ext=args.audio_ext,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            force_new_split=args.force_new_split,
        )

    return load_s2l_records(
        split=split,
        save_dir=save_dir,
        split_path=args.split_path,
        dataset_name=args.s2l_dataset_name,
        subset=args.s2l_subset,
        target_col=args.s2l_target_col,
        language=args.s2l_language,
        human_only=not args.s2l_include_tts,
        group_col=args.s2l_group_col,
        audio_col=args.s2l_audio_col,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        force_new_split=args.force_new_split,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
    )
