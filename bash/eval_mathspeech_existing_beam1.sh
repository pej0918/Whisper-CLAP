#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT="/home/pej0918/Projects/Audio_Text"
DATA_XLSX="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
AUDIO_DIR="/data1/eunju/datasets/mathspeech/dataset"
EXP_ROOT="${ROOT}/MathSpeech/Experiments"
SPLIT_PATH="${EXP_ROOT}/source_aware_seed42/split_indices.pt"
LOG_DIR="${EXP_ROOT}/logs_existing_beam1_eval"

BATCH_SIZE=4
NUM_WORKERS=2
NUM_BEAMS=1
MAX_NEW_TOKENS=256

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

run_metrics () {
  local dir="$1"
  local pred_col="$2"
  python compute_asr_metrics.py \
    --csv "${dir}/test_beam1.csv" \
    --pred_col "${pred_col}" \
    --ref_col transcription \
    --out_csv "${dir}/test_metrics_beam1.csv"
}

# 1. Whisper-base
(
  set -euo pipefail
  DIR="${EXP_ROOT}/epoch10_beam5_whisper_base"
  python eval_whisper.py \
    --model_kind hf \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --output_csv "${DIR}/test_beam1.csv" \
    --summary_json "${DIR}/summary_beam1.json" \
    --eval_split test \
    --pred_col pred_whisper_base_test_beam1 \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  run_metrics "${DIR}" pred_whisper_base_test_beam1
) > "${LOG_DIR}/01_whisper_base_beam1.log" 2>&1

# 2. Full fine-tuning
(
  set -euo pipefail
  DIR="${EXP_ROOT}/epoch10_beam5_whisper_full_ft"
  python eval_whisper.py \
    --model_kind hf \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${DIR}/best.pt" \
    --output_csv "${DIR}/test_beam1.csv" \
    --summary_json "${DIR}/summary_beam1.json" \
    --eval_split test \
    --pred_col pred_whisper_full_ft_test_beam1 \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  run_metrics "${DIR}" pred_whisper_full_ft_test_beam1
) > "${LOG_DIR}/02_full_finetuning_beam1.log" 2>&1

# 3. CLAP-guided full fine-tuning
(
  set -euo pipefail
  DIR="${EXP_ROOT}/epoch10_beam5_clap_guided_full_ft_align010"
  python eval_whisper.py \
    --model_kind clap_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${DIR}/best.pt" \
    --output_csv "${DIR}/test_beam1.csv" \
    --summary_json "${DIR}/summary_beam1.json" \
    --eval_split test \
    --pred_col pred_clap_guided_full_ft_test_beam1 \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  run_metrics "${DIR}" pred_clap_guided_full_ft_test_beam1
) > "${LOG_DIR}/03_clap_guided_full_ft_beam1.log" 2>&1

# 4. Ours
(
  set -euo pipefail
  DIR="${EXP_ROOT}/epoch10_beam5_clap_adapter_frozen_align010"
  python eval_whisper.py \
    --model_kind clap_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${DIR}/best.pt" \
    --output_csv "${DIR}/test_beam1.csv" \
    --summary_json "${DIR}/summary_beam1.json" \
    --eval_split test \
    --pred_col pred_ours_clap_adapter_test_beam1 \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  run_metrics "${DIR}" pred_ours_clap_adapter_test_beam1
) > "${LOG_DIR}/04_ours_clap_adapter_beam1.log" 2>&1

# 5. Controlled LoRA-Whisper
(
  set -euo pipefail
  DIR="${EXP_ROOT}/epoch10_beam5_lora_whisper_controlled_r32_qkvfc"
  python eval_whisper.py \
    --model_kind lora \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --adapter_dir "${DIR}/best_adapter" \
    --output_csv "${DIR}/test_beam1.csv" \
    --summary_json "${DIR}/summary_beam1.json" \
    --eval_split test \
    --pred_col pred_lora_whisper_controlled_test_beam1 \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  run_metrics "${DIR}" pred_lora_whisper_controlled_test_beam1
) > "${LOG_DIR}/05_lora_whisper_controlled_beam1.log" 2>&1

# 6. KAUST-style residual adapter
(
  set -euo pipefail
  DIR="${EXP_ROOT}/epoch10_beam5_residual_adapter_b256"
  python eval_whisper.py \
    --model_kind residual_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${DIR}/best.pt" \
    --output_csv "${DIR}/test_beam1.csv" \
    --summary_json "${DIR}/summary_beam1.json" \
    --eval_split test \
    --pred_col pred_residual_adapter_test_beam1 \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
  run_metrics "${DIR}" pred_residual_adapter_test_beam1
) > "${LOG_DIR}/06_residual_adapter_beam1.log" 2>&1

echo "[DONE] Existing-checkpoint beam1 eval finished. Logs: ${LOG_DIR}"
