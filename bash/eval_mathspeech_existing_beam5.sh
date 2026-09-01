#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT="/home/pej0918/Projects/Audio_Text"
DATA_XLSX="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
AUDIO_DIR="/data1/eunju/datasets/mathspeech/dataset"
EXP_ROOT="${ROOT}/MathSpeech/Experiments"
SPLIT_PATH="${EXP_ROOT}/source_aware_seed42/split_indices.pt"
LOG_DIR="${EXP_ROOT}/logs_eval_existing_beam5"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"

BATCH_SIZE=4
NUM_WORKERS=2
NUM_BEAMS=5
MAX_NEW_TOKENS=256

BASE_DIR="${EXP_ROOT}/epoch10_beam5_whisper_base"
FULL_FT_DIR="${EXP_ROOT}/epoch10_beam5_whisper_full_ft"
CLAP_FULL_DIR="${EXP_ROOT}/epoch10_beam5_clap_guided_full_ft_align010"
OURS_DIR="${EXP_ROOT}/epoch10_beam5_clap_adapter_frozen_align010"
LORA_DIR="${EXP_ROOT}/epoch10_beam5_lora_whisper_r16"
RESIDUAL_DIR="${EXP_ROOT}/epoch10_beam5_residual_adapter_b256"

run_eval_and_metrics () {
  local TAG="$1"
  local MODEL_KIND="$2"
  local OUT_DIR="$3"
  local PRED_COL="$4"
  local GPU="$5"
  local EXTRA_ARGS="$6"

  mkdir -p "${OUT_DIR}"
  echo "[${TAG}] eval on GPU ${GPU}"

  CUDA_VISIBLE_DEVICES="${GPU}" python eval_whisper.py \
    --model_kind "${MODEL_KIND}" \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --output_csv "${OUT_DIR}/test.csv" \
    --summary_json "${OUT_DIR}/summary.json" \
    --eval_split test \
    --pred_col "${PRED_COL}" \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${EXTRA_ARGS}

  python compute_asr_metrics.py \
    --csv "${OUT_DIR}/test.csv" \
    --pred_col "${PRED_COL}" \
    --ref_col transcription \
    --out_csv "${OUT_DIR}/test_metrics.csv"
}

(
  set -euo pipefail
  run_eval_and_metrics "BASE" "hf" "${BASE_DIR}" "pred_whisper_base_test" 0 ""
) > "${LOG_DIR}/01_whisper_base.log" 2>&1 &

(
  set -euo pipefail
  run_eval_and_metrics "FULL_FT" "hf" "${FULL_FT_DIR}" "pred_whisper_full_ft_test" 1 "--ckpt_path ${FULL_FT_DIR}/best.pt"
) > "${LOG_DIR}/02_full_finetuning.log" 2>&1 &

(
  set -euo pipefail
  run_eval_and_metrics "CLAP_FULL" "clap_adapter" "${CLAP_FULL_DIR}" "pred_clap_guided_full_ft_test" 2 "--ckpt_path ${CLAP_FULL_DIR}/best.pt"
) > "${LOG_DIR}/03_clap_guided_full_ft.log" 2>&1 &

(
  set -euo pipefail
  run_eval_and_metrics "OURS" "clap_adapter" "${OURS_DIR}" "pred_ours_clap_adapter_test" 3 "--ckpt_path ${OURS_DIR}/best.pt"
) > "${LOG_DIR}/04_ours_clap_adapter.log" 2>&1 &

if [ -d "${LORA_DIR}/best_adapter" ]; then
  (
    set -euo pipefail
    run_eval_and_metrics "LORA" "lora" "${LORA_DIR}" "pred_lora_whisper_test" 4 "--adapter_dir ${LORA_DIR}/best_adapter"
  ) > "${LOG_DIR}/05_lora_whisper.log" 2>&1 &
else
  echo "[SKIP] LoRA best_adapter not found: ${LORA_DIR}/best_adapter"
  echo "[SKIP] Fix peft/transformers version, then rerun LoRA training."
fi

(
  set -euo pipefail
  run_eval_and_metrics "RESIDUAL" "residual_adapter" "${RESIDUAL_DIR}" "pred_residual_adapter_test" 5 "--ckpt_path ${RESIDUAL_DIR}/best.pt"
) > "${LOG_DIR}/06_residual_adapter.log" 2>&1 &

echo "[INFO] Launched existing checkpoint eval jobs."
echo "[INFO] Logs: ${LOG_DIR}"
jobs -p
wait

echo "[DONE] Existing checkpoint beam5 eval finished."
bash "${ROOT}/bash/collect_mathspeech_beam5_results.sh"
