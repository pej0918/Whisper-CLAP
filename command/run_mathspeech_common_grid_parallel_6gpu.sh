#!/usr/bin/env bash
set -euo pipefail

# 6-GPU parallel MathSpeech grid on the team/common pipeline.
# Run from repository root:
#   GPUS="0 1 2 3 4 5" bash command/run_mathspeech_common_grid_parallel_6gpu.sh
#
# This launches independent train->eval jobs in parallel. Each trainable
# method/LR job owns one GPU and evaluates both beam=1 and beam=5 after its
# checkpoint is ready.

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
OURS_BATCH_SIZE="${OURS_BATCH_SIZE:-4}"
RESIDUAL_BATCH_SIZE="${RESIDUAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LRS=("1e-5" "1e-4")
BEAMS=("1" "5")
GPUS_LIST=(${GPUS:-0 1 2 3 4 5})
MAX_PARALLEL="${#GPUS_LIST[@]}"

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
TRAIN_CLAP_EMB="${TRAIN_CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_seed${SEED}}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"

mkdir -p "${OUT_ROOT}/logs"

printf '\n[CONFIG]\n'
echo "ROOT                : ${ROOT}"
echo "MODEL               : ${MODEL}"
echo "TRAIN_CSV           : ${TRAIN_CSV}"
echo "VAL_CSV             : ${VAL_CSV}"
echo "TEST_CSV            : ${TEST_CSV}"
echo "TRAIN_CLAP_EMB      : ${TRAIN_CLAP_EMB}"
echo "OUT_ROOT            : ${OUT_ROOT}"
echo "GPUS                : ${GPUS_LIST[*]}"
echo "MAX_PARALLEL        : ${MAX_PARALLEL}"
echo "EPOCHS              : ${EPOCHS}"
echo "LRS                 : ${LRS[*]}"
echo "BEAMS               : ${BEAMS[*]}"
echo "LORA_TARGET_MODULES : ${LORA_TARGET_MODULES}"

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing file: ${p}" >&2
    exit 1
  fi
}

require_file "${TRAIN_CSV}"
require_file "${VAL_CSV}"
require_file "${TEST_CSV}"
require_file "${TRAIN_CLAP_EMB}"

# -----------------------------
# Eval helpers
# -----------------------------
run_eval_whisper() {
  local gpu="$1"
  local model_path="$2"
  local out_dir="$3"
  local beam="$4"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_whisper.py \
    --manifest "${TEST_CSV}" \
    --model "${model_path}" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${beam}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_eval_ours() {
  local gpu="$1"
  local ckpt="$2"
  local out_dir="$3"
  local beam="$4"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_whisper_ours_m3av.py \
    --manifest "${TEST_CSV}" \
    --ckpt "${ckpt}" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${beam}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --fp16 \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_eval_residual() {
  local gpu="$1"
  local ckpt="$2"
  local out_dir="$3"
  local beam="$4"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
    --manifest "${TEST_CSV}" \
    --ckpt "${ckpt}" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --model "${MODEL}" \
    --adapter_bottleneck 256 \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${beam}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

# -----------------------------
# Job helpers
# -----------------------------
job_fullft() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/fullft_${lr_tag}"
  mkdir -p "${out_dir}"
  echo "[START] fullft ${lr} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_fullft.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --output_dir "${out_dir}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${lr}" \
    --train_batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do
    run_eval_whisper "${gpu}" "${out_dir}/best" "${out_dir}" "${beam}"
  done
  echo "[DONE] fullft ${lr} GPU=${gpu}"
}

job_ours() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/ours_${lr_tag}"
  mkdir -p "${out_dir}"
  echo "[START] ours ${lr} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_ours_m3av.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --train_clap_emb "${TRAIN_CLAP_EMB}" \
    --save_dir "${out_dir}" \
    --whisper_name "${MODEL}" \
    --adapter_type gated \
    --pool_type mean \
    --adapter_bottleneck 256 \
    --dropout 0.1 \
    --adapter_scale_init 0.01 \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --lambda_hidden 0.1 \
    --batch_size "${OURS_BATCH_SIZE}" \
    --eval_batch_size "${OURS_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${lr}" \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --eval_num_beams 5 \
    --generation_max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do
    run_eval_ours "${gpu}" "${out_dir}/best.pt" "${out_dir}" "${beam}"
  done
  echo "[DONE] ours ${lr} GPU=${gpu}"
}

job_lora() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/lora_whisper_${lr_tag}_outproj"
  mkdir -p "${out_dir}"
  echo "[START] lora ${lr} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_lora_controlled.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --output_dir "${out_dir}" \
    --model "${MODEL}" \
    --rank 32 \
    --lora_alpha 32 \
    --target_modules "${LORA_TARGET_MODULES}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${lr}" \
    --train_batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --num_workers "${NUM_WORKERS}" \
    --generation_num_beams 5 \
    --generation_max_length "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do
    run_eval_whisper "${gpu}" "${out_dir}/merged" "${out_dir}" "${beam}"
  done
  echo "[DONE] lora ${lr} GPU=${gpu}"
}

job_residual() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/residual_b256_${lr_tag}"
  mkdir -p "${out_dir}"
  echo "[START] residual ${lr} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --save_dir "${out_dir}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --lr "${lr}" \
    --batch_size "${RESIDUAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --adapter_bottleneck 256 \
    --selection_num_beams 5 \
    --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --fp16 \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do
    run_eval_residual "${gpu}" "${out_dir}/best.pt" "${out_dir}" "${beam}"
  done
  echo "[DONE] residual ${lr} GPU=${gpu}"
}

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]]; do
    wait -n
  done
}

launch() {
  local gpu="$1"
  local label="$2"
  shift 2
  echo "[LAUNCH] ${label} on GPU ${gpu}"
  "$@" "${gpu}" &
}

# Base eval is short; run once on GPU0 while no other jobs are running.
base_gpu="${GPUS_LIST[0]}"
for beam in "${BEAMS[@]}"; do
  run_eval_whisper "${base_gpu}" "${MODEL}" "${OUT_ROOT}/whisper_base" "${beam}"
done

# Launch 8 trainable jobs over available GPUs.
idx=0
for lr in "${LRS[@]}"; do
  for method in fullft ours lora residual; do
    wait_for_slot
    gpu="${GPUS_LIST[$((idx % MAX_PARALLEL))]}"
    case "${method}" in
      fullft) launch "${gpu}" "fullft_${lr}" job_fullft "${lr}" ;;
      ours) launch "${gpu}" "ours_${lr}" job_ours "${lr}" ;;
      lora) launch "${gpu}" "lora_${lr}" job_lora "${lr}" ;;
      residual) launch "${gpu}" "residual_${lr}" job_residual "${lr}" ;;
    esac
    idx=$((idx + 1))
  done
done

wait

"${PYTHON}" -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}"
echo "\n[DONE] Results collected under: ${OUT_ROOT}"
