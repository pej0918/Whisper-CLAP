#!/usr/bin/env bash
set -euo pipefail

# Full MathSpeech grid on the team/common pipeline.
# Run from repository root:
#   GPU=0 bash command/run_mathspeech_common_grid_lr_beam.sh
#
# Grid:
#   methods: whisper_base, fullft, clap_fullft, ours, lora_whisper, residual_b256
#   train LR for trainable methods: 1e-5, 1e-4
#   eval beams for every trained/evaluated method: 1, 5
#   epochs: 10
#
# Notes:
# - LoRA target modules default to the team-code setting including out_proj.
# - Selection for Seq2SeqTrainer methods follows the team Trainer recipe.
# - Checkpoints are trained once per LR and evaluated with both beam=1 and beam=5.

GPU="${GPU:-0}"
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
LRS=("${LRS:-1e-5 1e-4}")
BEAMS=("${BEAMS:-1 5}")

# Team manifest split paths. Override these env vars if needed.
SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
TRAIN_CLAP_EMB="${TRAIN_CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"

# Store experiment outputs under /data1/eunju by default.
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_seed${SEED}}"

# Team-code LoRA setting: includes out_proj.
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
echo "GPU                 : ${GPU}"
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

run_eval_whisper() {
  local model_path="$1"
  local out_dir="$2"
  local beam="$3"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper.py \
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
  local ckpt="$1"
  local out_dir="$2"
  local beam="$3"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper_ours_m3av.py \
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
  local ckpt="$1"
  local out_dir="$2"
  local beam="$3"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
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

# 0) No-training baseline: evaluate with every beam.
for beam in ${BEAMS[*]}; do
  run_eval_whisper "${MODEL}" "${OUT_ROOT}/whisper_base" "${beam}"
done

for lr in ${LRS[*]}; do
  lr_tag="lr${lr}"

  # 1) Full fine-tuning, team Seq2SeqTrainer code.
  full_dir="${OUT_ROOT}/fullft_${lr_tag}"
  mkdir -p "${full_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_fullft.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --output_dir "${full_dir}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${lr}" \
    --train_batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    2>&1 | tee "${full_dir}/train.log"
  for beam in ${BEAMS[*]}; do
    run_eval_whisper "${full_dir}/best" "${full_dir}" "${beam}"
  done

  # 2) CLAP-guided Full FT / Ours-style M3AV with trainable full Whisper is not
  # in the final team M3AV script. We therefore use the team's Ours script with
  # frozen Whisper below as the canonical CLAP-guided adapter method.

  # 3) Ours: team M3AV code, frozen Whisper, gated adapter + CLAP alignment.
  ours_dir="${OUT_ROOT}/ours_${lr_tag}"
  mkdir -p "${ours_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_ours_m3av.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --train_clap_emb "${TRAIN_CLAP_EMB}" \
    --save_dir "${ours_dir}" \
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
    2>&1 | tee "${ours_dir}/train.log"
  for beam in ${BEAMS[*]}; do
    run_eval_ours "${ours_dir}/best.pt" "${ours_dir}" "${beam}"
  done

  # 4) LoRA-Whisper: controlled script, team target modules including out_proj.
  lora_dir="${OUT_ROOT}/lora_whisper_${lr_tag}_outproj"
  mkdir -p "${lora_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_lora_controlled.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --output_dir "${lora_dir}" \
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
    2>&1 | tee "${lora_dir}/train.log"
  for beam in ${BEAMS[*]}; do
    run_eval_whisper "${lora_dir}/merged" "${lora_dir}" "${beam}"
  done

  # 5) KAUST-style Residual Adapter: frozen Whisper, encoder layer-wise residual adapter.
  residual_dir="${OUT_ROOT}/residual_b256_${lr_tag}"
  mkdir -p "${residual_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --save_dir "${residual_dir}" \
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
    2>&1 | tee "${residual_dir}/train.log"
  for beam in ${BEAMS[*]}; do
    run_eval_residual "${residual_dir}/best.pt" "${residual_dir}" "${beam}"
  done

done

"${PYTHON}" -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}"
echo "\n[DONE] Results collected under: ${OUT_ROOT}"
