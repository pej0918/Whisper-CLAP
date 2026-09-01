#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT="/home/pej0918/Projects/Audio_Text"
DATA_XLSX="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
AUDIO_DIR="/data1/eunju/datasets/mathspeech/dataset"
EXP_ROOT="${ROOT}/MathSpeech/Experiments"
SPLIT_PATH="${EXP_ROOT}/source_aware_seed42/split_indices.pt"
LOG_DIR="${EXP_ROOT}/logs_lora_residual_epoch10_beam5"
mkdir -p "${LOG_DIR}"

LORA_DIR="${EXP_ROOT}/epoch10_lora_whisper_r16_beam5"
RESIDUAL_DIR="${EXP_ROOT}/epoch10_residual_adapter_b256_beam5"
mkdir -p "${LORA_DIR}" "${RESIDUAL_DIR}"

cd "${ROOT}"

if [ ! -f "${SPLIT_PATH}" ]; then
  echo "[ERROR] split file not found: ${SPLIT_PATH}"
  echo "Run bash/run_mathspeech_4methods_epoch10.sh or create source_aware_seed42 split first."
  exit 1
fi

# ============================================================
# 1. LoRA-Whisper, CE only
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[1/2] LoRA-Whisper | GPU 1"

  CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${LORA_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed 42 \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj,v_proj \
    --num_workers 2

  CUDA_VISIBLE_DEVICES=1 python eval_whisper.py \
    --model_kind lora \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --adapter_dir "${LORA_DIR}/best_adapter" \
    --output_csv "${LORA_DIR}/test.csv" \
    --summary_json "${LORA_DIR}/test_summary.json" \
    --eval_split test \
    --pred_col pred_lora_whisper_test \
    --whisper_name openai/whisper-base \
    --num_beams 5 \
    --batch_size 16 \
    --num_workers 4

  python compute_asr_metrics.py \
    --csv "${LORA_DIR}/test.csv" \
    --pred_col pred_lora_whisper_test \
    --ref_col transcription \
    --out_csv "${LORA_DIR}/test_metrics.csv"

  echo "[1/2] Done"
) > "${LOG_DIR}/01_lora_whisper.log" 2>&1 &

# ============================================================
# 2. Residual Adapter-Whisper, CE only
# ============================================================
(
  set -euo pipefail
  cd "${ROOT}"
  echo "[2/2] Residual Adapter | GPU 2"

  CUDA_VISIBLE_DEVICES=2 python train_whisper_residual_adapter.py \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${RESIDUAL_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed 42 \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-5 \
    --adapter_bottleneck 256 \
    --num_workers 2

  CUDA_VISIBLE_DEVICES=2 python eval_whisper.py \
    --model_kind residual_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${RESIDUAL_DIR}/best.pt" \
    --output_csv "${RESIDUAL_DIR}/test.csv" \
    --summary_json "${RESIDUAL_DIR}/test_summary.json" \
    --eval_split test \
    --pred_col pred_residual_adapter_test \
    --whisper_name openai/whisper-base \
    --num_beams 5 \
    --batch_size 16 \
    --num_workers 4

  python compute_asr_metrics.py \
    --csv "${RESIDUAL_DIR}/test.csv" \
    --pred_col pred_residual_adapter_test \
    --ref_col transcription \
    --out_csv "${RESIDUAL_DIR}/test_metrics.csv"

  echo "[2/2] Done"
) > "${LOG_DIR}/02_residual_adapter.log" 2>&1 &

echo "[INFO] Launched MathSpeech LoRA/Residual jobs with beam=5 final eval."
echo "[INFO] Logs:"
echo "  ${LOG_DIR}/01_lora_whisper.log"
echo "  ${LOG_DIR}/02_residual_adapter.log"
jobs -p
wait

echo "[DONE] MathSpeech LoRA/Residual finished."
find "${LORA_DIR}" "${RESIDUAL_DIR}" -name "test_metrics.csv" -print -exec cat {} \;
find "${LORA_DIR}" "${RESIDUAL_DIR}" -name "test_summary.json" -print -exec cat {} \;
