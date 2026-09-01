#!/usr/bin/env bash
set -euo pipefail

# Fixed 6-GPU MathSpeech grid.
# GPU0: FullFT lr1e-5 -> Residual lr1e-5
# GPU1: FullFT lr1e-4 -> Residual lr1e-4
# GPU2: Ours lr1e-5
# GPU3: Ours lr1e-4
# GPU4: LoRA lr1e-5
# GPU5: LoRA lr1e-4
# Every checkpoint is evaluated with beam=1 and beam=5.

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
BEAMS=(1 5)

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
# FullFT/LoRA/Residual only need audio/text fields.
TRAIN_LORA_CSV="${TRAIN_LORA_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_LORA_CSV="${VAL_LORA_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
# Ours requires sample_id for CLAP embedding lookup.
TRAIN_MAIN_CSV="${TRAIN_MAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
VAL_MAIN_CSV="${VAL_MAIN_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
TRAIN_CLAP_EMB="${TRAIN_CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_seed${SEED}_6gpu}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"

mkdir -p "${OUT_ROOT}/logs"
for f in "${TRAIN_LORA_CSV}" "${VAL_LORA_CSV}" "${TRAIN_MAIN_CSV}" "${VAL_MAIN_CSV}" "${TEST_CSV}" "${TRAIN_CLAP_EMB}"; do
  [[ -f "$f" ]] || { echo "[ERROR] missing: $f" >&2; exit 1; }
done

printf '\n[CONFIG]\n'
echo "ROOT                : ${ROOT}"
echo "OUT_ROOT            : ${OUT_ROOT}"
echo "TRAIN_LORA_CSV      : ${TRAIN_LORA_CSV}"
echo "VAL_LORA_CSV        : ${VAL_LORA_CSV}"
echo "TRAIN_MAIN_CSV      : ${TRAIN_MAIN_CSV}"
echo "VAL_MAIN_CSV        : ${VAL_MAIN_CSV}"
echo "TEST_CSV            : ${TEST_CSV}"
echo "EPOCHS              : ${EPOCHS}"
echo "LORA_TARGET_MODULES : ${LORA_TARGET_MODULES}"
"${PYTHON}" - <<'PY'
import inspect, transformers
from transformers import Seq2SeqTrainingArguments
print('transformers:', transformers.__version__)
params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
print('warmup_ratio supported:', 'warmup_ratio' in params)
print('warmup_steps supported:', 'warmup_steps' in params)
PY

run_eval_whisper() {
  local gpu="$1" model_path="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON}" -u scripts/eval_whisper.py \
    --manifest "${TEST_CSV}" --model "$model_path" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --num_beams "$beam" --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_eval_ours() {
  local gpu="$1" ckpt="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON}" -u scripts/eval_whisper_ours_m3av.py \
    --manifest "${TEST_CSV}" --ckpt "$ckpt" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --num_beams "$beam" --max_new_tokens "${MAX_NEW_TOKENS}" --fp16 \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_eval_residual() {
  local gpu="$1" ckpt="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
    --manifest "${TEST_CSV}" --ckpt "$ckpt" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --model "${MODEL}" --adapter_bottleneck 256 \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --num_beams "$beam" --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_fullft() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/fullft_lr${lr}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON}" -u scripts/train_whisper_fullft_compat.py \
    --train "${TRAIN_LORA_CSV}" --dev "${VAL_LORA_CSV}" --output_dir "$out_dir" \
    --model "${MODEL}" --epochs "${EPOCHS}" --learning_rate "$lr" \
    --train_batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
    --generation_num_beams 5 --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_whisper "$gpu" "${out_dir}/best" "$out_dir" "$beam"; done
}

run_ours() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/ours_lr${lr}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON}" -u scripts/train_whisper_ours_m3av.py \
    --train "${TRAIN_MAIN_CSV}" --dev "${VAL_MAIN_CSV}" --train_clap_emb "${TRAIN_CLAP_EMB}" \
    --save_dir "$out_dir" --whisper_name "${MODEL}" \
    --adapter_type gated --pool_type mean --adapter_bottleneck 256 \
    --dropout 0.1 --adapter_scale_init 0.01 --align_loss_type cosine \
    --lambda_align 0.05 --lambda_hidden 0.1 \
    --batch_size "${OURS_BATCH_SIZE}" --eval_batch_size "${OURS_BATCH_SIZE}" \
    --epochs "${EPOCHS}" --lr "$lr" --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --eval_num_beams 5 --generation_max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_ours "$gpu" "${out_dir}/best.pt" "$out_dir" "$beam"; done
}

run_lora() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/lora_whisper_lr${lr}_outproj"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON}" -u scripts/train_whisper_lora_controlled_compat.py \
    --train "${TRAIN_LORA_CSV}" --dev "${VAL_LORA_CSV}" --output_dir "$out_dir" \
    --model "${MODEL}" --rank 32 --lora_alpha 32 --target_modules "${LORA_TARGET_MODULES}" \
    --epochs "${EPOCHS}" --learning_rate "$lr" \
    --train_batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
    --generation_num_beams 5 --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_whisper "$gpu" "${out_dir}/merged" "$out_dir" "$beam"; done
}

run_residual() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/residual_b256_lr${lr}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
    --train "${TRAIN_LORA_CSV}" --dev "${VAL_LORA_CSV}" --save_dir "$out_dir" \
    --model "${MODEL}" --epochs "${EPOCHS}" --lr "$lr" \
    --batch_size "${RESIDUAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --adapter_bottleneck 256 --selection_num_beams 5 \
    --selection_max_new_tokens "${MAX_NEW_TOKENS}" --seed "${SEED}" --fp16 \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_residual "$gpu" "${out_dir}/best.pt" "$out_dir" "$beam"; done
}

# Baseline eval first on GPU0.
mkdir -p "${OUT_ROOT}/whisper_base"
for beam in "${BEAMS[@]}"; do run_eval_whisper 0 "${MODEL}" "${OUT_ROOT}/whisper_base" "$beam"; done

# Six independent GPU workers. GPU0/1 each execute two sequential training jobs.
(
  run_fullft 0 1e-5
  run_residual 0 1e-5
) > "${OUT_ROOT}/logs/gpu0_fullft_residual_lr1e-5.log" 2>&1 & p0=$!
(
  run_fullft 1 1e-4
  run_residual 1 1e-4
) > "${OUT_ROOT}/logs/gpu1_fullft_residual_lr1e-4.log" 2>&1 & p1=$!
(run_ours 2 1e-5) > "${OUT_ROOT}/logs/gpu2_ours_lr1e-5.log" 2>&1 & p2=$!
(run_ours 3 1e-4) > "${OUT_ROOT}/logs/gpu3_ours_lr1e-4.log" 2>&1 & p3=$!
(run_lora 4 1e-5) > "${OUT_ROOT}/logs/gpu4_lora_lr1e-5.log" 2>&1 & p4=$!
(run_lora 5 1e-4) > "${OUT_ROOT}/logs/gpu5_lora_lr1e-4.log" 2>&1 & p5=$!

status=0
for p in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do
  wait "$p" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "[ERROR] One or more GPU workers failed. Check ${OUT_ROOT}/logs/*.log" >&2
  exit 1
fi

"${PYTHON}" -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}"
echo "[DONE] ${OUT_ROOT}"
