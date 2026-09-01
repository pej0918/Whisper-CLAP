#!/bin/bash
set -euo pipefail

EXP_ROOT="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments"
OFFICIAL_LORA_DIR="${EXP_ROOT}/epoch10_beam5_lora_whisper_official_r32_qkvfc"
OFFICIAL_LORA_LOG="${EXP_ROOT}/logs_lora_whisper_official_epoch10_beam5/lora_whisper_official.log"

bash /home/pej0918/Projects/Audio_Text/bash/collect_mathspeech_beam5_results.sh

python - <<'PY'
import json
import re
from pathlib import Path

import pandas as pd

EXP_ROOT = Path('/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments')
d = EXP_ROOT / 'epoch10_beam5_lora_whisper_official_r32_qkvfc'
log = EXP_ROOT / 'logs_lora_whisper_official_epoch10_beam5' / 'lora_whisper_official.log'

params = 'PENDING'
if log.exists():
    text = log.read_text(errors='ignore')
    vals = re.findall(r'trainable params:\s*([0-9,]+)', text)
    if vals:
        raw = int(vals[-1].replace(',', ''))
        params = f'{raw / 1_000_000:.2f}M' if raw >= 1_000_000 else f'{raw / 1000:.2f}K'

best_valid = 'PENDING'
tl = d / 'train_log.csv'
if tl.exists():
    tdf = pd.read_csv(tl)
    if 'valid_wer' in tdf.columns and len(tdf):
        best_valid = f'{float(tdf["valid_wer"].min()):.4f}'

wer = cer = n = ins = dele = avglen = 'PENDING'
mcsv = d / 'test_metrics.csv'
if mcsv.exists():
    m = pd.read_csv(mcsv).iloc[0].to_dict()
    wer = f'{float(m.get("WER")):.4f}' if m.get('WER') is not None else 'PENDING'
    cer = f'{float(m.get("CER")):.4f}' if m.get('CER') is not None else 'PENDING'
    n = m.get('num_eval', 'PENDING')
    ins = m.get('total_insertions', 'PENDING')
    dele = m.get('total_deletions', 'PENDING')
    avglen = f'{float(m.get("AvgLenRatio")):.4f}' if m.get('AvgLenRatio') is not None else 'PENDING'

rtf = 'PENDING'
sj = d / 'summary.json'
if sj.exists():
    s = json.loads(sj.read_text())
    if s.get('rtf') is not None:
        rtf = f'{float(s["rtf"]):.4f}'

status = 'DONE' if wer != 'PENDING' and cer != 'PENDING' else ('CHECK_LOG' if log.exists() and 'Traceback' in log.read_text(errors='ignore')[-4000:] else 'RUNNING_OR_PENDING')

row = pd.DataFrame([{
    'Method': 'LoRA-Whisper official-style',
    'Status': status,
    'Trainable Params': params,
    'MathSpeech WER': wer,
    'MathSpeech CER': cer,
    'Best Valid WER': best_valid,
    'Num Eval': n,
    'Insertions': ins,
    'Deletions': dele,
    'AvgLenRatio': avglen,
    'RTF': rtf,
    'Setting': 'LoRA-Whisper paper-style; r=32; q/k/v+fc; lr=1e-4; epoch=10; beam=5; HF Whisper-base backbone',
}])
print('\n[Official-style LoRA result]')
print(row.to_markdown(index=False))
PY
