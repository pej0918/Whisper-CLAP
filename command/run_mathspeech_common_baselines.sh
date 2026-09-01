#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
#   bash command/run_mathspeech_common_baselines.sh

GPU="${GPU:-0}"
ROOT="${ROOT:-$PWD}"
SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/common_mathspeech}"
MODEL="${MODEL:-openai/whisper-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
RESIDUAL_BATCH_SIZE="${RESIDUAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_BEAMS="${NUM_BEAMS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,fc1,fc2}"

mkdir -p "${OUT_ROOT}" \
  "${OUT_ROOT}/whisper_base" \
  "${OUT_ROOT}/lora_fair_lr1e-5" \
  "${OUT_ROOT}/lora_paper_lr1e-4" \
  "${OUT_ROOT}/residual_b256"

printf '\n[CONFIG]\n'
echo "ROOT        : ${ROOT}"
echo "TRAIN_CSV   : ${TRAIN_CSV}"
echo "VAL_CSV     : ${VAL_CSV}"
echo "TEST_CSV    : ${TEST_CSV}"
echo "OUT_ROOT    : ${OUT_ROOT}"
echo "GPU         : ${GPU}"
echo "MODEL       : ${MODEL}"
echo "EPOCHS      : ${EPOCHS}"
echo "BEAM        : ${NUM_BEAMS}"
echo "TARGETS     : ${TARGET_MODULES}"

# 0) Whisper-base eval
CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval_whisper.py \
  --manifest "${TEST_CSV}" \
  --model "${MODEL}" \
  --output_csv "${OUT_ROOT}/whisper_base/test_predictions_beam5.csv" \
  --summary_json "${OUT_ROOT}/whisper_base/test_summary_beam5.json" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee "${OUT_ROOT}/whisper_base/eval.log"

# 1) LoRA-Whisper fair setting: lr=1e-5
CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/train_whisper_lora_controlled.py \
  --train "${TRAIN_CSV}" \
  --dev "${VAL_CSV}" \
  --output_dir "${OUT_ROOT}/lora_fair_lr1e-5" \
  --model "${MODEL}" \
  --rank 32 \
  --lora_alpha 32 \
  --target_modules "${TARGET_MODULES}" \
  --epochs "${EPOCHS}" \
  --learning_rate 1e-5 \
  --train_batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --num_workers "${NUM_WORKERS}" \
  --generation_num_beams "${NUM_BEAMS}" \
  --generation_max_length "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_ROOT}/lora_fair_lr1e-5/train.log"

CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval_whisper.py \
  --manifest "${TEST_CSV}" \
  --model "${OUT_ROOT}/lora_fair_lr1e-5/merged" \
  --output_csv "${OUT_ROOT}/lora_fair_lr1e-5/test_predictions_beam5.csv" \
  --summary_json "${OUT_ROOT}/lora_fair_lr1e-5/test_summary_beam5.json" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee "${OUT_ROOT}/lora_fair_lr1e-5/eval.log"

# 2) LoRA-Whisper paper LR setting: lr=1e-4
CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/train_whisper_lora_controlled.py \
  --train "${TRAIN_CSV}" \
  --dev "${VAL_CSV}" \
  --output_dir "${OUT_ROOT}/lora_paper_lr1e-4" \
  --model "${MODEL}" \
  --rank 32 \
  --lora_alpha 32 \
  --target_modules "${TARGET_MODULES}" \
  --epochs "${EPOCHS}" \
  --learning_rate 1e-4 \
  --train_batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --num_workers "${NUM_WORKERS}" \
  --generation_num_beams "${NUM_BEAMS}" \
  --generation_max_length "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_ROOT}/lora_paper_lr1e-4/train.log"

CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval_whisper.py \
  --manifest "${TEST_CSV}" \
  --model "${OUT_ROOT}/lora_paper_lr1e-4/merged" \
  --output_csv "${OUT_ROOT}/lora_paper_lr1e-4/test_predictions_beam5.csv" \
  --summary_json "${OUT_ROOT}/lora_paper_lr1e-4/test_summary_beam5.json" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee "${OUT_ROOT}/lora_paper_lr1e-4/eval.log"

# 3) KAUST-style Residual Adapter: frozen Whisper, layer-wise encoder residual adapter, b=256
CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/train_whisper_residual_adapter.py \
  --train "${TRAIN_CSV}" \
  --dev "${VAL_CSV}" \
  --save_dir "${OUT_ROOT}/residual_b256" \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --lr 1e-5 \
  --batch_size "${RESIDUAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --adapter_bottleneck 256 \
  --selection_num_beams "${NUM_BEAMS}" \
  --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  --fp16 \
  2>&1 | tee "${OUT_ROOT}/residual_b256/train.log"

CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval_whisper_residual_adapter.py \
  --manifest "${TEST_CSV}" \
  --ckpt "${OUT_ROOT}/residual_b256/best.pt" \
  --output_csv "${OUT_ROOT}/residual_b256/test_predictions_beam5.csv" \
  --summary_json "${OUT_ROOT}/residual_b256/test_summary_beam5.json" \
  --model "${MODEL}" \
  --adapter_bottleneck 256 \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee "${OUT_ROOT}/residual_b256/eval.log"

python -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}"
