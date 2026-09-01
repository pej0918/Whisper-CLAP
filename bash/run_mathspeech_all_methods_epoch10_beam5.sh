#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT="/home/pej0918/Projects/Audio_Text"
DATA_XLSX="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
AUDIO_DIR="/data1/eunju/datasets/mathspeech/dataset"
CLAP_EMB="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt"
EXP_ROOT="${ROOT}/MathSpeech/Experiments"
SPLIT_DIR="${EXP_ROOT}/source_aware_seed42"
SPLIT_PATH="${SPLIT_DIR}/split_indices.pt"
LOG_DIR="${EXP_ROOT}/logs_all_methods_epoch10_beam5"

mkdir -p "${SPLIT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

# ============================================================
# Common settings
# ============================================================
EPOCHS=10
LR="1e-5"
BATCH_SIZE=4
NUM_WORKERS=2
SEED=42
NUM_BEAMS=5
MAX_NEW_TOKENS=256
SELECTION_MAX_NEW_TOKENS=256

BASE_DIR="${EXP_ROOT}/epoch10_beam5_whisper_base"
FULL_FT_DIR="${EXP_ROOT}/epoch10_beam5_whisper_full_ft"
CLAP_FULL_DIR="${EXP_ROOT}/epoch10_beam5_clap_guided_full_ft_align010"
OURS_DIR="${EXP_ROOT}/epoch10_beam5_clap_adapter_frozen_align010"
LORA_DIR="${EXP_ROOT}/epoch10_beam5_lora_whisper_r16"
RESIDUAL_DIR="${EXP_ROOT}/epoch10_beam5_residual_adapter_b256"

mkdir -p \
  "${BASE_DIR}" \
  "${FULL_FT_DIR}" \
  "${CLAP_FULL_DIR}" \
  "${OURS_DIR}" \
  "${LORA_DIR}" \
  "${RESIDUAL_DIR}"

# ============================================================
# 0. Create or verify source-aware split, seed 42
#    This also trains/evals nothing; it just creates split_indices.pt if missing.
# ============================================================
if [ ! -f "${SPLIT_PATH}" ]; then
  echo "[INFO] Creating source-aware split: ${SPLIT_PATH}"
  python - <<PY
import os
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

# ============================================================
# 1. Whisper-base, no training
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[1/6] Whisper-base | GPU 0"

  CUDA_VISIBLE_DEVICES=0 python eval_whisper.py \
    --model_kind hf \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --output_csv "${BASE_DIR}/test.csv" \
    --summary_json "${BASE_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_whisper_base_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${BASE_DIR}/test.csv" \
    --pred_col pred_whisper_base_test \
    --ref_col transcription \
    --out_csv "${BASE_DIR}/test_metrics.csv"

  echo "[1/6] Done"
) > "${LOG_DIR}/01_whisper_base.log" 2>&1 &

# ============================================================
# 2. Full Fine-tuning, CE only
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[2/6] Full Fine-tuning | GPU 1"

  CUDA_VISIBLE_DEVICES=1 python train_whisper_ft.py \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${FULL_FT_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num_workers "${NUM_WORKERS}" \
    --selection_max_new_tokens "${SELECTION_MAX_NEW_TOKENS}"

  CUDA_VISIBLE_DEVICES=1 python eval_whisper.py \
    --model_kind hf \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${FULL_FT_DIR}/best.pt" \
    --output_csv "${FULL_FT_DIR}/test.csv" \
    --summary_json "${FULL_FT_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_whisper_full_ft_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${FULL_FT_DIR}/test.csv" \
    --pred_col pred_whisper_full_ft_test \
    --ref_col transcription \
    --out_csv "${FULL_FT_DIR}/test_metrics.csv"

  echo "[2/6] Done"
) > "${LOG_DIR}/02_full_finetuning.log" 2>&1 &

# ============================================================
# 3. CLAP-guided Full Fine-tuning
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[3/6] CLAP-guided Full FT | GPU 2"

  CUDA_VISIBLE_DEVICES=2 python train_whisper_clap_adapter.py \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --clap_emb_path "${CLAP_EMB}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${CLAP_FULL_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --adapter_type gated \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.10 \
    --lambda_hidden 0.10 \
    --num_workers "${NUM_WORKERS}" \
    --selection_max_new_tokens "${SELECTION_MAX_NEW_TOKENS}"

  CUDA_VISIBLE_DEVICES=2 python eval_whisper.py \
    --model_kind clap_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${CLAP_FULL_DIR}/best.pt" \
    --output_csv "${CLAP_FULL_DIR}/test.csv" \
    --summary_json "${CLAP_FULL_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_clap_guided_full_ft_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${CLAP_FULL_DIR}/test.csv" \
    --pred_col pred_clap_guided_full_ft_test \
    --ref_col transcription \
    --out_csv "${CLAP_FULL_DIR}/test_metrics.csv"

  echo "[3/6] Done"
) > "${LOG_DIR}/03_clap_guided_full_ft.log" 2>&1 &

# ============================================================
# 4. Ours: CLAP-guided Adapter, Whisper frozen
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[4/6] Ours CLAP Adapter | GPU 3"

  CUDA_VISIBLE_DEVICES=3 python train_whisper_clap_adapter.py \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --clap_emb_path "${CLAP_EMB}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${OURS_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --adapter_type gated \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.10 \
    --lambda_hidden 0.10 \
    --freeze_whisper \
    --num_workers "${NUM_WORKERS}" \
    --selection_max_new_tokens "${SELECTION_MAX_NEW_TOKENS}"

  CUDA_VISIBLE_DEVICES=3 python eval_whisper.py \
    --model_kind clap_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${OURS_DIR}/best.pt" \
    --output_csv "${OURS_DIR}/test.csv" \
    --summary_json "${OURS_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_ours_clap_adapter_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${OURS_DIR}/test.csv" \
    --pred_col pred_ours_clap_adapter_test \
    --ref_col transcription \
    --out_csv "${OURS_DIR}/test_metrics.csv"

  echo "[4/6] Done"
) > "${LOG_DIR}/04_ours_clap_adapter.log" 2>&1 &

# ============================================================
# 5. LoRA-Whisper, CE only
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[5/6] LoRA-Whisper | GPU 4"

  CUDA_VISIBLE_DEVICES=4 python train_whisper_lora.py \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${LORA_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj,v_proj \
    --num_workers "${NUM_WORKERS}" \
    --selection_max_new_tokens "${SELECTION_MAX_NEW_TOKENS}"

  CUDA_VISIBLE_DEVICES=4 python eval_whisper.py \
    --model_kind lora \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --adapter_dir "${LORA_DIR}/best_adapter" \
    --output_csv "${LORA_DIR}/test.csv" \
    --summary_json "${LORA_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_lora_whisper_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${LORA_DIR}/test.csv" \
    --pred_col pred_lora_whisper_test \
    --ref_col transcription \
    --out_csv "${LORA_DIR}/test_metrics.csv"

  echo "[5/6] Done"
) > "${LOG_DIR}/05_lora_whisper.log" 2>&1 &

# ============================================================
# 6. Residual Adapter-Whisper, CE only
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[6/6] Residual Adapter | GPU 5"

  CUDA_VISIBLE_DEVICES=5 python train_whisper_residual_adapter.py \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${RESIDUAL_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --adapter_bottleneck 256 \
    --num_workers "${NUM_WORKERS}" \
    --selection_max_new_tokens "${SELECTION_MAX_NEW_TOKENS}"

  CUDA_VISIBLE_DEVICES=5 python eval_whisper.py \
    --model_kind residual_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${RESIDUAL_DIR}/best.pt" \
    --output_csv "${RESIDUAL_DIR}/test.csv" \
    --summary_json "${RESIDUAL_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_residual_adapter_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${RESIDUAL_DIR}/test.csv" \
    --pred_col pred_residual_adapter_test \
    --ref_col transcription \
    --out_csv "${RESIDUAL_DIR}/test_metrics.csv"

  echo "[6/6] Done"
) > "${LOG_DIR}/06_residual_adapter.log" 2>&1 &

PIDS=$(jobs -p)
echo "[INFO] Launched MathSpeech all-method jobs."
echo "[INFO] PIDs:"
echo "${PIDS}"
echo "[INFO] Logs: ${LOG_DIR}"

wait

echo "[DONE] All MathSpeech methods finished."
echo "[DONE] Summary JSON files:"
find "${EXP_ROOT}" -maxdepth 2 -path "*epoch10_beam5*" -name "summary.json" -print -exec cat {} \;

echo "[DONE] Metric CSV files:"
find "${EXP_ROOT}" -maxdepth 2 -path "*epoch10_beam5*" -name "test_metrics.csv" -print -exec cat {} \;
