#!/usr/bin/env bash
set -euo pipefail

# Parallel 6-GPU MathSpeech grid on the team/common pipeline.
# Run from repository root:
#   GPUS="0 1 2 3 4 5" bash command/run_mathspeech_common_grid_lr_beam_parallel_6gpu.sh
#
# This script launches independent train+eval jobs in parallel:
#   fullft lr=1e-5, 1e-4
#   ours lr=1e-5, 1e-4
#   lora_whisper lr=1e-5, 1e-4
#   residual_b256 lr=1e-5, 1e-4
# plus whisper-base eval.
# Each trained checkpoint is evaluated with beam=1 and beam=5.

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
GPUS_STR="${GPUS:-0 1 2 3 4 5}"
read -r -a GPU_LIST <<< "${GPUS_STR}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_LIST[@]}}"
BEAMS_STR="${BEAMS:-1 5}"
read -r -a BEAMS <<< "${BEAMS_STR}"

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
TRAIN_CLAP_EMB="${TRAIN_CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_seed${SEED}_6gpu}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"

mkdir -p "${OUT_ROOT}/logs"

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

printf '\n[CONFIG]\n'
echo "ROOT                : ${ROOT}"
echo "MODEL               : ${MODEL}"
echo "TRAIN_CSV           : ${TRAIN_CSV}"
echo "VAL_CSV             : ${VAL_CSV}"
echo "TEST_CSV            : ${TEST_CSV}"
echo "TRAIN_CLAP_EMB      : ${TRAIN_CLAP_EMB}"
echo "OUT_ROOT            : ${OUT_ROOT}"
echo "GPUS                : ${GPU_LIST[*]}"
echo "MAX_PARALLEL        : ${MAX_PARALLEL}"
echo "EPOCHS              : ${EPOCHS}"
echo "BEAMS               : ${BEAMS[*]}"
echo "LORA_TARGET_MODULES : ${LORA_TARGET_MODULES}"

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

run_fullft_job() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/fullft_${lr_tag}"
  mkdir -p "${out_dir}"
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
}

run_ours_job() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/ours_${lr_tag}"
  mkdir -p "${out_dir}"
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
}

run_lora_job() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/lora_whisper_${lr_tag}_outproj"
  mkdir -p "${out_dir}"
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
}

run_residual_job() {
  local gpu="$1"
  local lr="$2"
  local lr_tag="lr${lr}"
  local out_dir="${OUT_ROOT}/residual_b256_${lr_tag}"
  mkdir -p "${out_dir}"
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
}

# Small no-training baseline first on GPU0.
for beam in "${BEAMS[@]}"; do
  run_eval_whisper "${GPU_LIST[0]}" "${MODEL}" "${OUT_ROOT}/whisper_base" "${beam}"
done

# Eight independent training jobs. Run first six simultaneously on GPUs 0-5,
# then the remaining two as soon as the first wave finishes. This avoids
# oversubscribing a single GPU and keeps the script simple/reproducible.
pids=()
job_names=()

launch_job() {
  local name="$1"
  shift
  echo "[LAUNCH] ${name}: $*"
  ( "$@" ) > "${OUT_ROOT}/logs/${name}.launcher.log" 2>&1 &
  pids+=("$!")
  job_names+=("${name}")
}

launch_job "fullft_lr1e-5_gpu${GPU_LIST[0]}" run_fullft_job "${GPU_LIST[0]}" "1e-5"
launch_job "fullft_lr1e-4_gpu${GPU_LIST[1]}" run_fullft_job "${GPU_LIST[1]}" "1e-4"
launch_job "ours_lr1e-5_gpu${GPU_LIST[2]}" run_ours_job "${GPU_LIST[2]}" "1e-5"
launch_job "ours_lr1e-4_gpu${GPU_LIST[3]}" run_ours_job "${GPU_LIST[3]}" "1e-4"
launch_job "lora_lr1e-5_gpu${GPU_LIST[4]}" run_lora_job "${GPU_LIST[4]}" "1e-5"
launch_job "lora_lr1e-4_gpu${GPU_LIST[5]}" run_lora_job "${GPU_LIST[5]}" "1e-4"

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[DONE] ${job_names[$i]}"
  else
    echo "[FAILED] ${job_names[$i]}" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[ERROR] First wave failed. Check ${OUT_ROOT}/logs/*.launcher.log" >&2
  exit "${status}"
fi

# Second wave: residual lr grid on GPUs 0 and 1.
pids=()
job_names=()
launch_job "residual_lr1e-5_gpu${GPU_LIST[0]}" run_residual_job "${GPU_LIST[0]}" "1e-5"
launch_job "residual_lr1e-4_gpu${GPU_LIST[1]}" run_residual_job "${GPU_LIST[1]}" "1e-4"

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[DONE] ${job_names[$i]}"
  else
    echo "[FAILED] ${job_names[$i]}" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[ERROR] Second wave failed. Check ${OUT_ROOT}/logs/*.launcher.log" >&2
  exit "${status}"
fi

"${PYTHON}" -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}"
echo "\n[DONE] Results collected under: ${OUT_ROOT}"
