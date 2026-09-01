#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT="/home/pej0918/Projects/Audio_Text"
DATA_XLSX="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
AUDIO_DIR="/data1/eunju/datasets/mathspeech/dataset"
EXP_ROOT="${ROOT}/MathSpeech/Experiments"
SPLIT_DIR="${EXP_ROOT}/source_aware_seed42"
SPLIT_PATH="${SPLIT_DIR}/split_indices.pt"
SAVE_DIR="${EXP_ROOT}/epoch10_beam5_lora_whisper_official_r32_qkvfc"
LOG_DIR="${EXP_ROOT}/logs_lora_whisper_official_epoch10_beam5"
LOG_FILE="${LOG_DIR}/lora_whisper_official.log"

mkdir -p "${SPLIT_DIR}" "${SAVE_DIR}" "${LOG_DIR}"
cd "${ROOT}"

EPOCHS=10
LR="1e-4"
BATCH_SIZE=4
NUM_WORKERS=2
SEED=42
NUM_BEAMS=5
MAX_NEW_TOKENS=256
SELECTION_MAX_NEW_TOKENS=256

if [ ! -f "${SPLIT_PATH}" ]; then
  echo "[INFO] Creating source-aware split: ${SPLIT_PATH}"
  python - <<PY
import pandas as pd
from mathspeech_utils import make_or_load_source_aware_split

df = pd.read_excel("${DATA_XLSX}")
make_or_load_source_aware_split(
    df=df,
    save_dir="${SPLIT_DIR}",
    split_path="${SPLIT_PATH}",
    source_col="Source",
    seed=${SEED},
    train_ratio=0.8,
    valid_ratio=0.1,
    force_new=False,
)
print("saved split:", "${SPLIT_PATH}")
PY
else
  echo "[INFO] Using existing split: ${SPLIT_PATH}"
fi

{
  echo "[LoRA-Whisper official-style] GPU=${CUDA_VISIBLE_DEVICES:-default}"
  echo "save_dir=${SAVE_DIR}"
  echo "settings: r=32 alpha=32 dropout=0.0 target=q_proj,k_proj,v_proj,fc1,fc2 lr=1e-4 epochs=10 beam=5"

  python train_whisper_lora.py \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --target_modules q_proj,k_proj,v_proj,fc1,fc2 \
    --num_workers "${NUM_WORKERS}" \
    --selection_max_new_tokens "${SELECTION_MAX_NEW_TOKENS}"

  python eval_whisper.py \
    --model_kind lora \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --adapter_dir "${SAVE_DIR}/best_adapter" \
    --output_csv "${SAVE_DIR}/test.csv" \
    --summary_json "${SAVE_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_lora_whisper_official_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${SAVE_DIR}/test.csv" \
    --pred_col pred_lora_whisper_official_test \
    --ref_col transcription \
    --out_csv "${SAVE_DIR}/test_metrics.csv"

  echo "[DONE] LoRA-Whisper official-style finished"
  echo "summary=${SAVE_DIR}/summary.json"
  echo "metrics=${SAVE_DIR}/test_metrics.csv"
} 2>&1 | tee "${LOG_FILE}"
