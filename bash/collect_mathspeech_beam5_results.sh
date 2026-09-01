#!/bin/bash
set -euo pipefail

ROOT="/home/pej0918/Projects/Audio_Text"
EXP_ROOT="${ROOT}/MathSpeech/Experiments"
LOG_DIR="${EXP_ROOT}/logs_all_methods_epoch10_beam5"
OUT_CSV="${EXP_ROOT}/mathspeech_beam5_results.csv"
OUT_MD="${EXP_ROOT}/mathspeech_beam5_results.md"

python - <<'PY'
import json
import os
import re
from pathlib import Path

import pandas as pd

EXP_ROOT = Path("/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments")
LOG_DIR = EXP_ROOT / "logs_all_methods_epoch10_beam5"
OUT_CSV = EXP_ROOT / "mathspeech_beam5_results.csv"
OUT_MD = EXP_ROOT / "mathspeech_beam5_results.md"

METHODS = [
    {
        "method": "Whisper-base",
        "dir": EXP_ROOT / "epoch10_beam5_whisper_base",
        "log": LOG_DIR / "01_whisper_base.log",
        "trainable_params_default": 0,
        "setting": "no training; beam=5 eval",
    },
    {
        "method": "Full Fine-tuning",
        "dir": EXP_ROOT / "epoch10_beam5_whisper_full_ft",
        "log": LOG_DIR / "02_full_finetuning.log",
        "setting": "CE only; lr=1e-5; valid WER-best; beam=5 eval",
    },
    {
        "method": "CLAP-guided Full Fine-tuning",
        "dir": EXP_ROOT / "epoch10_beam5_clap_guided_full_ft_align010",
        "log": LOG_DIR / "03_clap_guided_full_ft.log",
        "setting": "CE+align+hidden; lr=1e-5; lambda_align=0.10; lambda_hidden=0.10; beam=5 eval",
    },
    {
        "method": "Ours",
        "dir": EXP_ROOT / "epoch10_beam5_clap_adapter_frozen_align010",
        "log": LOG_DIR / "04_ours_clap_adapter.log",
        "setting": "Whisper frozen; CE+align+hidden; lr=1e-5; lambda_align=0.10; lambda_hidden=0.10; beam=5 eval",
    },
    {
        "method": "LoRA-Whisper",
        "dir": EXP_ROOT / "epoch10_beam5_lora_whisper_r16",
        "log": LOG_DIR / "05_lora_whisper.log",
        "setting": "LoRA r=16 alpha=32 q/v; CE only; lr=1e-5; valid WER-best; beam=5 eval",
    },
    {
        "method": "Residual Adapter",
        "dir": EXP_ROOT / "epoch10_beam5_residual_adapter_b256",
        "log": LOG_DIR / "06_residual_adapter.log",
        "setting": "Whisper frozen; residual adapter bottleneck=256; CE only; lr=1e-5; valid WER-best; beam=5 eval",
    },
]


def human_params(x):
    if x is None or x == "":
        return "PENDING"
    try:
        x = int(float(x))
    except Exception:
        return str(x)
    if x == 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x / 1_000:.2f}K"
    return str(x)


def read_trainable_params(log_path, default=None):
    if default is not None:
        return default
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"trainable params:\s*([0-9,]+)", text)
    if not matches:
        matches = re.findall(r"trainable_params\s*[=:]\s*([0-9,]+)", text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def read_metric_csv(path):
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return {}
        row = df.iloc[0].to_dict()
        return row
    except Exception as exc:
        return {"metric_error": str(exc)}


def read_summary_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"summary_error": str(exc)}


def last_log_status(log_path):
    if not log_path.exists():
        return "NOT_STARTED"
    text = log_path.read_text(errors="ignore")[-4000:]
    if "Done" in text or "FINAL RESULT" in text or "saved metric to" in text:
        return "DONE_OR_EVAL_DONE"
    if "Traceback" in text or "Error" in text or "Exception" in text:
        return "CHECK_LOG"
    return "RUNNING_OR_PENDING"


rows = []
for item in METHODS:
    d = item["dir"]
    metrics = read_metric_csv(d / "test_metrics.csv")
    summary = read_summary_json(d / "summary.json")
    params = read_trainable_params(item["log"], item.get("trainable_params_default"))

    wer = metrics.get("WER", summary.get("wer", None))
    cer = metrics.get("CER", summary.get("cer", None))
    valid_wer = None

    # Try training log csv first; otherwise grep plain log.
    train_log_csv = d / "train_log.csv"
    if train_log_csv.exists():
        try:
            tdf = pd.read_csv(train_log_csv)
            if "valid_wer" in tdf.columns and len(tdf) > 0:
                valid_wer = float(tdf["valid_wer"].min())
        except Exception:
            pass
    if valid_wer is None and item["log"].exists():
        text = item["log"].read_text(errors="ignore")
        vals = re.findall(r"best_valid_wer:\s*([0-9.]+)", text)
        if vals:
            valid_wer = float(vals[-1])

    status = "DONE" if (wer is not None and cer is not None) else last_log_status(item["log"])
    rows.append({
        "Method": item["method"],
        "Status": status,
        "Trainable Params": human_params(params),
        "Trainable Params Raw": params,
        "MathSpeech WER": wer if wer is not None else "PENDING",
        "MathSpeech CER": cer if cer is not None else "PENDING",
        "Best Valid WER": valid_wer if valid_wer is not None else "PENDING",
        "Num Eval": metrics.get("num_eval", summary.get("samples", "PENDING")),
        "Insertions": metrics.get("total_insertions", "PENDING"),
        "Deletions": metrics.get("total_deletions", "PENDING"),
        "AvgLenRatio": metrics.get("AvgLenRatio", "PENDING"),
        "RTF": summary.get("rtf", "PENDING"),
        "Dir": str(d),
        "Log": str(item["log"]),
        "Setting": item["setting"],
    })

out = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False)

show_cols = [
    "Method", "Status", "Trainable Params", "MathSpeech WER", "MathSpeech CER",
    "Best Valid WER", "Num Eval", "Insertions", "Deletions", "AvgLenRatio", "RTF", "Setting"
]
show = out[show_cols].copy()

for col in ["MathSpeech WER", "MathSpeech CER", "Best Valid WER", "AvgLenRatio", "RTF"]:
    def fmt(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)
    show[col] = show[col].map(fmt)

md = show.to_markdown(index=False)
OUT_MD.write_text(md + "\n")

print("=" * 120)
print(md)
print("=" * 120)
print("saved csv:", OUT_CSV)
print("saved md :", OUT_MD)
PY
