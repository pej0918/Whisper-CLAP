#!/usr/bin/env bash
set -euo pipefail

# Low-level single-configuration launcher for the finalized MathSpeech protocol.
#
# Final common protocol:
#   backbone             = openai/whisper-base
#   max epochs           = 10
#   LR search            = {1e-5, 1e-4, 3e-4}
#   checkpoint selection = lowest validation WER @ beam=5 within each LR
#   LR selection         = lowest validation WER across the three LRs
#   test decoding        = beam=5, only after LR selection
#   lambda_hidden        = 0.1 (CLAP-guided methods)
#   lambda_align         = 0.05 (CLAP-guided methods)
#
# STAGE controls whether this file trains, tests, or does both:
#   STAGE=train : train one METHOD x LR; do not touch the test set
#   STAGE=test  : evaluate an already-trained METHOD x LR on test
#   STAGE=full  : train then test (useful for fixed-LR ablations)
#
# For the full 3-LR search + validation-based selection, prefer:
#   command/run_mathspeech_final_protocol.sh

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
STAGE="${STAGE:-full}"

case "${STAGE}" in
  train|test|full) ;;
  *) echo "[ERROR] STAGE must be train, test, or full; got ${STAGE}" >&2; exit 2 ;;
esac

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
OURS_TRAIN_CSV="${OURS_TRAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
OURS_VAL_CSV="${OURS_VAL_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
OURS_TEST_CSV="${OURS_TEST_CSV:-${SPLIT_DIR}/mathspeech_test_source_seed42.csv}"
CLAP_EMB="${CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_final_protocol_seed${SEED}}"

mkdir -p "${OUT_ROOT}"

# These defaults are the validation-selected LRs from the finalized
# {1e-5, 1e-4, 3e-4} search on MathSpeech. During an LR sweep, always pass LR explicitly.
case "${METHOD}" in
  whisper_base) DEFAULT_LR="none" ;;
  fullft) DEFAULT_LR="1e-5" ;;
  clap_fullft) DEFAULT_LR="1e-5" ;;
  lora_noout) DEFAULT_LR="3e-4" ;;
  lora_out) DEFAULT_LR="1e-4" ;;
  residual_b256) DEFAULT_LR="1e-4" ;;
  ours) DEFAULT_LR="3e-4" ;;
  rq4_ce_only) DEFAULT_LR="3e-4" ;;
  rq4_ce_hidden) DEFAULT_LR="3e-4" ;;
  rq4_ce_align) DEFAULT_LR="3e-4" ;;
  rq5b_lora_r7) DEFAULT_LR="3e-4" ;;
  rq5b_residual_b128) DEFAULT_LR="3e-4" ;;
  *)
    echo "[ERROR] Unknown METHOD=${METHOD}" >&2
    echo "Allowed: whisper_base fullft clap_fullft lora_noout lora_out residual_b256 ours rq4_ce_only rq4_ce_hidden rq4_ce_align rq5b_lora_r7 rq5b_residual_b128" >&2
    exit 2
    ;;
esac

if [[ "${METHOD}" == "whisper_base" ]]; then
  LR="none"
  OUT_DIR="${OUT_ROOT}/${METHOD}"
else
  LR="${LR:-${DEFAULT_LR}}"
  case "${LR}" in
    1e-5|1e-4|3e-4) ;;
    *) echo "[ERROR] Final LR search space is {1e-5, 1e-4, 3e-4}; got LR=${LR}" >&2; exit 2 ;;
  esac
  OUT_DIR="${OUT_ROOT}/${METHOD}_lr${LR}"
fi
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

run_eval_residual() {
  local bottleneck="$1"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
    --manifest "${TEST_CSV}" \
    --ckpt "${OUT_DIR}/best.pt" \
    --model "${MODEL}" \
    --adapter_bottleneck "${bottleneck}" \
    --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
    --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${TEST_BEAM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
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
}

train_method() {
  case "${METHOD}" in
    whisper_base)
      echo "[INFO] whisper_base has no training stage"
      ;;
    fullft)
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_fullft_compat.py \
        --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${OUT_DIR}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --learning_rate "${LR}" --train_batch_size 16 --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
        --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}"
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
      ;;
    residual_b256)
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
        --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --save_dir "${OUT_DIR}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --lr "${LR}" --batch_size 4 --num_workers "${NUM_WORKERS}" \
        --adapter_bottleneck 256 --selection_num_beams "${SELECTION_BEAM}" \
        --selection_max_new_tokens "${MAX_NEW_TOKENS}" --seed "${SEED}" --fp16
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
      ;;
    rq5b_residual_b128)
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
        --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --save_dir "${OUT_DIR}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --lr "${LR}" --batch_size 4 --num_workers "${NUM_WORKERS}" \
        --adapter_bottleneck 128 --selection_num_beams "${SELECTION_BEAM}" \
        --selection_max_new_tokens "${MAX_NEW_TOKENS}" --seed "${SEED}" --fp16
      ;;
  esac
}

test_method() {
  case "${METHOD}" in
    whisper_base) run_eval_whisper "${MODEL}" ;;
    fullft) run_eval_whisper "${OUT_DIR}/best" ;;
    clap_fullft|ours|rq4_ce_only|rq4_ce_hidden|rq4_ce_align) run_eval_projector ;;
    lora_noout|lora_out|rq5b_lora_r7) run_eval_whisper "${OUT_DIR}/merged" ;;
    residual_b256) run_eval_residual 256 ;;
    rq5b_residual_b128) run_eval_residual 128 ;;
  esac
}

echo "METHOD=${METHOD} STAGE=${STAGE} GPU=${GPU} LR=${LR} OUT_DIR=${OUT_DIR}"

if [[ "${STAGE}" == "train" || "${STAGE}" == "full" ]]; then
  train_method
fi
if [[ "${STAGE}" == "test" || "${STAGE}" == "full" ]]; then
  test_method
fi

echo "DONE: METHOD=${METHOD} STAGE=${STAGE} OUT_DIR=${OUT_DIR}"
