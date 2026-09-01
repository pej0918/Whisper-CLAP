#!/usr/bin/env bash
set -euo pipefail

# Fixed 6-GPU MathSpeech grid.
# - FullFT/LoRA/Residual use the team's lora-format manifests.
# - Ours uses the source-balanced manifests containing sample_id.
# - FullFT uses a transformers-version-compatible training wrapper.
# - LoRA controlled script is transformers-version compatible.
# - Every trained checkpoint: epoch=10, lr in {1e-5,1e-4}, eval beam in {1,5}.
#
# Run:
#   GPUS="0 1 2 3 4 5" bash command/run_mathspeech_common_grid_lr_beam_parallel_6gpu_v2.sh

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
BEAMS=(1 5)

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_LORA_CSV="${TRAIN_LORA_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_LORA_CSV="${VAL_LORA_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
OURS_TRAIN_CSV="${OURS_TRAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
OURS_VAL_CSV="${OURS_VAL_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
TRAIN_CLAP_EMB="${TRAIN_CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_seed${SEED}_6gpu_v2}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"

mkdir -p "${OUT_ROOT}/logs"

for f in "${TRAIN_LORA_CSV}" "${VAL_LORA_CSV}" "${OURS_TRAIN_CSV}" "${OURS_VAL_CSV}" "${TEST_CSV}" "${TRAIN_CLAP_EMB}"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing: $f" >&2; exit 1; }
done

printf '\n[CONFIG]\n'
echo "ROOT                : ${ROOT}"
echo "OUT_ROOT            : ${OUT_ROOT}"
echo "GPUS                : ${GPU_LIST[*]}"
echo "EPOCHS              : ${EPOCHS}"
echo "LR                  : 1e-5, 1e-4"
echo "EVAL BEAMS          : 1, 5"
echo "FULLFT/LORA TRAIN   : ${TRAIN_LORA_CSV}"
echo "OURS TRAIN          : ${OURS_TRAIN_CSV}"
echo "LORA TARGETS        : ${LORA_TARGET_MODULES}"

run_eval_whisper() {
  local gpu="$1" model_path="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_whisper.py \
    --manifest "$TEST_CSV" --model "$model_path" \
    --output_csv "$out_dir/test_predictions_beam${beam}.csv" \
    --summary_json "$out_dir/test_summary_beam${beam}.json" \
    --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --num_beams "$beam" --max_new_tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "$out_dir/eval_beam${beam}.log"
}

run_eval_ours() {
  local gpu="$1" ckpt="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_whisper_ours_m3av.py \
    --manifest "$TEST_CSV" --ckpt "$ckpt" \
    --output_csv "$out_dir/test_predictions_beam${beam}.csv" \
    --summary_json "$out_dir/test_summary_beam${beam}.json" \
    --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --num_beams "$beam" --max_new_tokens "$MAX_NEW_TOKENS" --fp16 \
    2>&1 | tee "$out_dir/eval_beam${beam}.log"
}

run_eval_residual() {
  local gpu="$1" ckpt="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_whisper_residual_adapter.py \
    --manifest "$TEST_CSV" --ckpt "$ckpt" --model "$MODEL" \
    --adapter_bottleneck 256 \
    --output_csv "$out_dir/test_predictions_beam${beam}.csv" \
    --summary_json "$out_dir/test_summary_beam${beam}.json" \
    --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --num_beams "$beam" --max_new_tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "$out_dir/eval_beam${beam}.log"
}

run_fullft() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/fullft_lr${lr}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_whisper_fullft_compat.py \
    --train "$TRAIN_LORA_CSV" --dev "$VAL_LORA_CSV" --output_dir "$out_dir" \
    --model "$MODEL" --epochs "$EPOCHS" --learning_rate "$lr" \
    --train_batch_size "$BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps 1 --num_workers "$NUM_WORKERS" \
    --generation_num_beams 5 --generation_max_length "$MAX_NEW_TOKENS" --seed "$SEED" \
    2>&1 | tee "$out_dir/train.log"
  for b in "${BEAMS[@]}"; do run_eval_whisper "$gpu" "$out_dir/best" "$out_dir" "$b"; done
}

run_ours() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/ours_lr${lr}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_whisper_ours_m3av.py \
    --train "$OURS_TRAIN_CSV" --dev "$OURS_VAL_CSV" --train_clap_emb "$TRAIN_CLAP_EMB" \
    --save_dir "$out_dir" --whisper_name "$MODEL" \
    --adapter_type gated --pool_type mean --adapter_bottleneck 256 \
    --dropout 0.1 --adapter_scale_init 0.01 \
    --align_loss_type cosine --lambda_align 0.05 --lambda_hidden 0.1 \
    --batch_size "$OURS_BATCH_SIZE" --eval_batch_size "$OURS_BATCH_SIZE" \
    --epochs "$EPOCHS" --lr "$lr" --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 --num_workers "$NUM_WORKERS" --seed "$SEED" \
    --eval_num_beams 5 --generation_max_new_tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "$out_dir/train.log"
  for b in "${BEAMS[@]}"; do run_eval_ours "$gpu" "$out_dir/best.pt" "$out_dir" "$b"; done
}

run_lora() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/lora_whisper_lr${lr}_outproj"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_whisper_lora_controlled.py \
    --train "$TRAIN_LORA_CSV" --dev "$VAL_LORA_CSV" --output_dir "$out_dir" \
    --model "$MODEL" --rank 32 --lora_alpha 32 --target_modules "$LORA_TARGET_MODULES" \
    --epochs "$EPOCHS" --learning_rate "$lr" \
    --train_batch_size "$BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps 1 --num_workers "$NUM_WORKERS" \
    --generation_num_beams 5 --generation_max_length "$MAX_NEW_TOKENS" --seed "$SEED" \
    2>&1 | tee "$out_dir/train.log"
  for b in "${BEAMS[@]}"; do run_eval_whisper "$gpu" "$out_dir/merged" "$out_dir" "$b"; done
}

run_residual() {
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/residual_b256_lr${lr}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_whisper_residual_adapter.py \
    --train "$TRAIN_LORA_CSV" --dev "$VAL_LORA_CSV" --save_dir "$out_dir" \
    --model "$MODEL" --epochs "$EPOCHS" --lr "$lr" \
    --batch_size "$RESIDUAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
    --adapter_bottleneck 256 --selection_num_beams 5 \
    --selection_max_new_tokens "$MAX_NEW_TOKENS" --seed "$SEED" --fp16 \
    2>&1 | tee "$out_dir/train.log"
  for b in "${BEAMS[@]}"; do run_eval_residual "$gpu" "$out_dir/best.pt" "$out_dir" "$b"; done
}

# Base: one checkpoint, both decoding settings.
mkdir -p "${OUT_ROOT}/whisper_base"
for b in "${BEAMS[@]}"; do run_eval_whisper "${GPU_LIST[0]}" "$MODEL" "${OUT_ROOT}/whisper_base" "$b"; done

launch() {
  local name="$1"; shift
  echo "[LAUNCH] $name"
  ( "$@" ) > "${OUT_ROOT}/logs/${name}.launcher.log" 2>&1 &
  echo $!
}

# First wave: fill GPUs 0-5.
pids=()
names=()
for spec in \
  "fullft_1e-5:${GPU_LIST[0]}:run_fullft:1e-5" \
  "fullft_1e-4:${GPU_LIST[1]}:run_fullft:1e-4" \
  "ours_1e-5:${GPU_LIST[2]}:run_ours:1e-5" \
  "ours_1e-4:${GPU_LIST[3]}:run_ours:1e-4" \
  "lora_1e-5:${GPU_LIST[4]}:run_lora:1e-5" \
  "lora_1e-4:${GPU_LIST[5]}:run_lora:1e-4"; do
  IFS=: read -r n g fn lr <<< "$spec"
  names+=("$n")
  pids+=("$(launch "$n" "$fn" "$g" "$lr")")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[DONE] ${names[$i]}"; else echo "[FAILED] ${names[$i]}" >&2; status=1; fi
done
[[ $status -eq 0 ]] || { echo "[ERROR] First wave failed. Check ${OUT_ROOT}/logs" >&2; exit 1; }

# Second wave: residual LR grid.
pids=("$(launch residual_1e-5 run_residual "${GPU_LIST[0]}" 1e-5)" \
      "$(launch residual_1e-4 run_residual "${GPU_LIST[1]}" 1e-4)")
names=(residual_1e-5 residual_1e-4)
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[DONE] ${names[$i]}"; else echo "[FAILED] ${names[$i]}" >&2; status=1; fi
done
[[ $status -eq 0 ]] || { echo "[ERROR] Residual wave failed. Check ${OUT_ROOT}/logs" >&2; exit 1; }

"$PYTHON" -u scripts/collect_asr_results.py --output_dir "$OUT_ROOT"
echo "[DONE] ${OUT_ROOT}"
