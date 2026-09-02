#!/usr/bin/env bash
set -euo pipefail

# Canonical single-GPU launcher for the MathSpeech configurations used in the
# current paper tables. Checkpoints are selected within each run by the lowest
# validation WER (beam=5). The default learning rate is method-specific and can
# be overridden with LR=... for controlled comparisons.
#
# Examples:
#   METHOD=ours GPU=1 bash command/run_mathspeech_reported_configs.sh
#   METHOD=lora_noout GPU=2 bash command/run_mathspeech_reported_configs.sh
#   METHOD=rq4_ce_align GPU=3 bash command/run_mathspeech_reported_configs.sh
#   METHOD=rq5b_lora_r7 LR=3e-4 GPU=4 bash command/run_mathspeech_reported_configs.sh

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
METHOD="${METHOD:-ours}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
SELECTION_BEAM="${SELECTION_BEAM:-5}"
TEST_BEAM="${TEST_BEAM:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
OURS_TRAIN_CSV="${OURS_TRAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
OURS_VAL_CSV="${OURS_VAL_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
OURS_TEST_CSV="${OURS_TEST_CSV:-${SPLIT_DIR}/mathspeech_test_source_seed42.csv}"
CLAP_EMB="${CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_reported_configs_seed${SEED}}"

mkdir -p "${OUT_ROOT}"

case "${METHOD}" in
  whisper_base) DEFAULT_LR="none" ;;
  fullft) DEFAULT_LR="1e-5" ;;
  clap_fullft) DEFAULT_LR="1e-5" ;;
  lora_noout) DEFAULT_LR="5e-4" ;;
  lora_out) DEFAULT_LR="5e-4" ;;
  residual_b256) DEFAULT_LR="5e-4" ;;
  ours) DEFAULT_LR="3e-4" ;;
  rq4_ce_only) DEFAULT_LR="3e-4" ;;
  rq4_ce_hidden) DEFAULT_LR="3e-4" ;;
  rq4_ce_align) DEFAULT_LR="3e-4" ;;
  rq5b_lora_r7) DEFAULT_LR="5e-4" ;;
  rq5b_residual_b128) DEFAULT_LR="5e-4" ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    echo "Allowed: whisper_base fullft clap_fullft lora_noout lora_out residual_b256 ours rq4_ce_only rq4_ce_hidden rq4_ce_align rq5b_lora_r7 rq5b_residual_b128" >&2
    exit 2
    ;;
esac

LR="${LR:-${DEFAULT_LR}}"
OUT_DIR="${OUT_ROOT}/${METHOD}${LR:+_lr${LR}}"
mkdir -p "${OUT_DIR}"

run_eval_whisper() {
  local model_path="$1"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper.py \
    --manifest "${TEST_CSV}" \
    --model "${model_path}" \
    --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
    --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${TEST_BEAM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
}

run_eval_projector() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_mathspeech_projector_source_disjoint_hf.py \
    --manifest "${OURS_TEST_CSV}" \
    --ckpt "${OUT_DIR}/best.pt" \
    --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
    --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${TEST_BEAM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --fp16
}

run_projector_train() {
  local freeze_whisper="$1"
  local lambda_hidden="$2"
  local lambda_align="$3"
  local align_type="$4"
  local freeze_args=()
  [[ "${freeze_whisper}" == "true" ]] && freeze_args+=(--freeze_whisper)

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_mathspeech_projector_best_wer.py \
    --train_csv "${OURS_TRAIN_CSV}" \
    --valid_csv "${OURS_VAL_CSV}" \
    --test_csv "${OURS_TEST_CSV}" \
    --save_dir "${OUT_DIR}" \
    --clap_emb_path "${CLAP_EMB}" \
    --whisper_name "${MODEL}" \
    --adapter_type gated \
    --pool_type cls \
    --adapter_bottleneck 256 \
    --dropout 0.1 \
    --adapter_scale_init 0.01 \
    --align_loss_type "${align_type}" \
    --lambda_align "${lambda_align}" \
    --lambda_hidden "${lambda_hidden}" \
    --batch_size 4 \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay 1e-4 \
    --selection_num_beams "${SELECTION_BEAM}" \
    --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    "${freeze_args[@]}"

  run_eval_projector
}

echo "METHOD=${METHOD} GPU=${GPU} LR=${LR} OUT_DIR=${OUT_DIR}"

case "${METHOD}" in
  whisper_base)
    run_eval_whisper "${MODEL}"
    ;;

  fullft)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_fullft_compat.py \
      --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${OUT_DIR}" --model "${MODEL}" \
      --epochs "${EPOCHS}" --learning_rate "${LR}" --train_batch_size 16 --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
      --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}"
    run_eval_whisper "${OUT_DIR}/best"
    ;;

  clap_fullft)
    run_projector_train false 0.1 0.05 cosine
    ;;

  lora_noout|lora_out)
    if [[ "${METHOD}" == "lora_out" ]]; then
      TARGETS="q_proj,k_proj,v_proj,out_proj,fc1,fc2"
    else
      TARGETS="q_proj,k_proj,v_proj,fc1,fc2"
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_lora_controlled.py \
      --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${OUT_DIR}" --model "${MODEL}" \
      --rank 32 --lora_alpha 32 --target_modules "${TARGETS}" \
      --epochs "${EPOCHS}" --learning_rate "${LR}" --train_batch_size 16 --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
      --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}"
    run_eval_whisper "${OUT_DIR}/merged"
    ;;

  residual_b256)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
      --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --save_dir "${OUT_DIR}" --model "${MODEL}" \
      --epochs "${EPOCHS}" --lr "${LR}" --batch_size 4 --num_workers "${NUM_WORKERS}" \
      --adapter_bottleneck 256 --selection_num_beams "${SELECTION_BEAM}" \
      --selection_max_new_tokens "${MAX_NEW_TOKENS}" --seed "${SEED}" --fp16
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
      --manifest "${TEST_CSV}" --ckpt "${OUT_DIR}/best.pt" --model "${MODEL}" --adapter_bottleneck 256 \
      --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
      --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
      --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
      --num_beams "${TEST_BEAM}" --max_new_tokens "${MAX_NEW_TOKENS}"
    ;;

  ours)
    run_projector_train true 0.1 0.05 cosine
    ;;

  rq4_ce_only)
    run_projector_train true 0.0 0.0 none
    ;;

  rq4_ce_hidden)
    run_projector_train true 0.1 0.0 none
    ;;

  rq4_ce_align)
    run_projector_train true 0.0 0.05 cosine
    ;;

  rq5b_lora_r7)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_lora_controlled.py \
      --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${OUT_DIR}" --model "${MODEL}" \
      --rank 7 --lora_alpha 7 --target_modules "q_proj,k_proj,v_proj,fc1,fc2" \
      --epochs "${EPOCHS}" --learning_rate "${LR}" --train_batch_size 16 --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
      --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}"
    run_eval_whisper "${OUT_DIR}/merged"
    ;;

  rq5b_residual_b128)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
      --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --save_dir "${OUT_DIR}" --model "${MODEL}" \
      --epochs "${EPOCHS}" --lr "${LR}" --batch_size 4 --num_workers "${NUM_WORKERS}" \
      --adapter_bottleneck 128 --selection_num_beams "${SELECTION_BEAM}" \
      --selection_max_new_tokens "${MAX_NEW_TOKENS}" --seed "${SEED}" --fp16
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
      --manifest "${TEST_CSV}" --ckpt "${OUT_DIR}/best.pt" --model "${MODEL}" --adapter_bottleneck 128 \
      --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
      --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
      --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
      --num_beams "${TEST_BEAM}" --max_new_tokens "${MAX_NEW_TOKENS}"
    ;;
esac

echo "DONE: ${OUT_DIR}"
