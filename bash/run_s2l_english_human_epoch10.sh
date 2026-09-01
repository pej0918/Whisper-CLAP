#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT="/home/pej0918/Projects/Audio_Text"
EXP_ROOT="${ROOT}/Speech2Latex/Experiments"
LOG_DIR="${EXP_ROOT}/logs_english_human_epoch10_beam5"
mkdir -p "${LOG_DIR}"

cd "${ROOT}"

run_one_subset () {
  local SUBSET="$1"      # sentences or equations
  local TAG="$2"         # s2l_sent or s2l_eq
  local GPU_BASE="$3"
  local GPU_FT="$4"
  local GPU_LORA="$5"
  local GPU_RES="$6"

  local SUB_ROOT="${EXP_ROOT}/${TAG}_beam5"
  local SPLIT_PATH="${SUB_ROOT}/split.pt"
  local BASE_DIR="${SUB_ROOT}/whisper_base"
  local FT_DIR="${SUB_ROOT}/whisper_full_ft"
  local LORA_DIR="${SUB_ROOT}/lora_whisper_r16"
  local RES_DIR="${SUB_ROOT}/residual_adapter_b256"
  mkdir -p "${BASE_DIR}" "${FT_DIR}" "${LORA_DIR}" "${RES_DIR}"

  echo "[INFO] Running ${SUBSET} English Human subset with final eval beam=5"

  # 1. Whisper-base eval only
  (
    set -euo pipefail
    cd "${ROOT}"
    echo "[${TAG} BASE] GPU ${GPU_BASE}"
    CUDA_VISIBLE_DEVICES=${GPU_BASE} python eval_whisper.py \
      --model_kind hf \
      --dataset_type s2l \
      --s2l_subset "${SUBSET}" \
      --s2l_language eng \
      --s2l_target_col pronunciation \
      --split_path "${SPLIT_PATH}" \
      --force_new_split \
      --output_csv "${BASE_DIR}/test.csv" \
      --summary_json "${BASE_DIR}/test_summary.json" \
      --eval_split test \
      --pred_col pred_whisper_base_test \
      --whisper_name openai/whisper-base \
      --num_beams 5 \
      --batch_size 16 \
      --num_workers 4

    python compute_asr_metrics.py \
      --csv "${BASE_DIR}/test.csv" \
      --pred_col pred_whisper_base_test \
      --ref_col transcription \
      --out_csv "${BASE_DIR}/test_metrics.csv"
  ) > "${LOG_DIR}/${TAG}_01_base.log" 2>&1 &

  # 2. Full FT, CE only
  (
    set -euo pipefail
    cd "${ROOT}"
    echo "[${TAG} FULL_FT] GPU ${GPU_FT}"
    CUDA_VISIBLE_DEVICES=${GPU_FT} python train_whisper_ft_generic.py \
      --dataset_type s2l \
      --s2l_subset "${SUBSET}" \
      --s2l_language eng \
      --s2l_target_col pronunciation \
      --split_path "${SPLIT_PATH}" \
      --save_dir "${FT_DIR}" \
      --whisper_name openai/whisper-base \
      --seed 42 \
      --epochs 10 \
      --batch_size 4 \
      --lr 1e-5 \
      --num_workers 2

    CUDA_VISIBLE_DEVICES=${GPU_FT} python eval_whisper.py \
      --model_kind hf \
      --dataset_type s2l \
      --s2l_subset "${SUBSET}" \
      --s2l_language eng \
      --s2l_target_col pronunciation \
      --split_path "${SPLIT_PATH}" \
      --ckpt_path "${FT_DIR}/best.pt" \
      --output_csv "${FT_DIR}/test.csv" \
      --summary_json "${FT_DIR}/test_summary.json" \
      --eval_split test \
      --pred_col pred_whisper_full_ft_test \
      --whisper_name openai/whisper-base \
      --num_beams 5 \
      --batch_size 16 \
      --num_workers 4

    python compute_asr_metrics.py \
      --csv "${FT_DIR}/test.csv" \
      --pred_col pred_whisper_full_ft_test \
      --ref_col transcription \
      --out_csv "${FT_DIR}/test_metrics.csv"
  ) > "${LOG_DIR}/${TAG}_02_full_ft.log" 2>&1 &

  # 3. LoRA-Whisper, CE only
  (
    set -euo pipefail
    cd "${ROOT}"
    echo "[${TAG} LORA] GPU ${GPU_LORA}"
    CUDA_VISIBLE_DEVICES=${GPU_LORA} python train_whisper_lora.py \
      --dataset_type s2l \
      --s2l_subset "${SUBSET}" \
      --s2l_language eng \
      --s2l_target_col pronunciation \
      --split_path "${SPLIT_PATH}" \
      --save_dir "${LORA_DIR}" \
      --whisper_name openai/whisper-base \
      --seed 42 \
      --epochs 10 \
      --batch_size 4 \
      --lr 1e-5 \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --target_modules q_proj,v_proj \
      --num_workers 2

    CUDA_VISIBLE_DEVICES=${GPU_LORA} python eval_whisper.py \
      --model_kind lora \
      --dataset_type s2l \
      --s2l_subset "${SUBSET}" \
      --s2l_language eng \
      --s2l_target_col pronunciation \
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
  ) > "${LOG_DIR}/${TAG}_03_lora.log" 2>&1 &

  # 4. Residual Adapter, CE only
  (
    set -euo pipefail
    cd "${ROOT}"
    echo "[${TAG} RESIDUAL] GPU ${GPU_RES}"
    CUDA_VISIBLE_DEVICES=${GPU_RES} python train_whisper_residual_adapter.py \
      --dataset_type s2l \
      --s2l_subset "${SUBSET}" \
      --s2l_language eng \
      --s2l_target_col pronunciation \
      --split_path "${SPLIT_PATH}" \
      --save_dir "${RES_DIR}" \
      --whisper_name openai/whisper-base \
      --seed 42 \
      --epochs 10 \
      --batch_size 4 \
      --lr 1e-5 \
      --adapter_bottleneck 256 \
      --num_workers 2

    CUDA_VISIBLE_DEVICES=${GPU_RES} python eval_whisper.py \
      --model_kind residual_adapter \
      --dataset_type s2l \
      --s2l_subset "${SUBSET}" \
      --s2l_language eng \
      --s2l_target_col pronunciation \
      --split_path "${SPLIT_PATH}" \
      --ckpt_path "${RES_DIR}/best.pt" \
      --output_csv "${RES_DIR}/test.csv" \
      --summary_json "${RES_DIR}/test_summary.json" \
      --eval_split test \
      --pred_col pred_residual_adapter_test \
      --whisper_name openai/whisper-base \
      --num_beams 5 \
      --batch_size 16 \
      --num_workers 4

    python compute_asr_metrics.py \
      --csv "${RES_DIR}/test.csv" \
      --pred_col pred_residual_adapter_test \
      --ref_col transcription \
      --out_csv "${RES_DIR}/test_metrics.csv"
  ) > "${LOG_DIR}/${TAG}_04_residual.log" 2>&1 &
}

# Run S2L-sentences on GPUs 1-4.
run_one_subset sentences s2l_sent 1 2 3 4
wait

# Then run S2L-equations on GPUs 1-4.
run_one_subset equations s2l_eq 1 2 3 4
wait

echo "[DONE] S2L English Human experiments finished."
find "${EXP_ROOT}" -path "*beam5*" -name "test_metrics.csv" -print -exec cat {} \;
find "${EXP_ROOT}" -path "*beam5*" -name "test_summary.json" -print -exec cat {} \;
