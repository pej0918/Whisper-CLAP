#!/usr/bin/env bash
set -euo pipefail

# Reuse already-completed MathSpeech baseline results and rerun only:
#   - Ours: lr={1e-5,1e-4}
#   - CLAP-guided Full Fine-tuning: lr={1e-5,1e-4}
# with BEST VALIDATION WER checkpoint selection (selection beam=5).
#
# Usage:
#   GPUS="0 1 2 3" bash command/rerun_mathspeech_projector_bestvalidwer_4gpu.sh

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
OURS_BATCH_SIZE="${OURS_BATCH_SIZE:-4}"
CLAP_FULLFT_BATCH_SIZE="${CLAP_FULLFT_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SELECTION_BEAM="${SELECTION_BEAM:-5}"
BEAMS_STR="${BEAMS:-1 5}"
read -r -a BEAMS <<< "${BEAMS_STR}"

GPUS_STR="${GPUS:-0 1 2 3}"
read -r -a GPU_LIST <<< "${GPUS_STR}"
if [[ "${#GPU_LIST[@]}" -lt 4 ]]; then
  echo "[ERROR] Need at least 4 GPUs, got: ${GPU_LIST[*]}" >&2
  exit 1
fi

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/mathspeech_test_source_seed42.csv}"
CLAP_EMB="${CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"

OLD_ROOT="${OLD_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_seed${SEED}_6gpu}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_bestvalidwer_seed${SEED}_6gpu}"

require_file(){ [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
require_dir(){ [[ -d "$1" ]] || { echo "[ERROR] Missing dir: $1" >&2; exit 1; }; }
for f in "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}" "${CLAP_EMB}"; do require_file "$f"; done
require_dir "${OLD_ROOT}"

mkdir -p "${OUT_ROOT}/logs"

echo "============================================================"
echo "MathSpeech projector rerun: BEST VALIDATION WER"
echo "OLD_ROOT          = ${OLD_ROOT}"
echo "OUT_ROOT          = ${OUT_ROOT}"
echo "GPUS              = ${GPU_LIST[*]}"
echo "EPOCHS            = ${EPOCHS}"
echo "LRs               = 1e-5, 1e-4"
echo "selection metric  = validation WER"
echo "selection beam    = ${SELECTION_BEAM}"
echo "test beams        = ${BEAMS[*]}"
echo "============================================================"

# Copy methods that were already selected by validation WER.
BASELINE_DIRS=(
  whisper_base
  fullft_lr1e-5
  fullft_lr1e-4
  lora_whisper_lr1e-5_outproj
  lora_whisper_lr1e-4_outproj
  lora_whisper_lr1e-5_nooutproj
  lora_whisper_lr1e-4_nooutproj
  residual_b256_lr1e-5
  residual_b256_lr1e-4
)

for name in "${BASELINE_DIRS[@]}"; do
  require_dir "${OLD_ROOT}/${name}"
  if [[ ! -d "${OUT_ROOT}/${name}" ]]; then
    echo "[COPY] ${name}"
    cp -a "${OLD_ROOT}/${name}" "${OUT_ROOT}/${name}"
  else
    echo "[KEEP] ${OUT_ROOT}/${name} already exists"
  fi
done

run_eval_projector(){
  local gpu="$1"
  local ckpt="$2"
  local out_dir="$3"
  local beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_mathspeech_projector_source_disjoint_hf.py \
    --manifest "${TEST_CSV}" \
    --ckpt "$ckpt" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "$beam" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --fp16 \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_projector_job(){
  local gpu="$1"
  local lr="$2"
  local freeze="$3"
  local out_dir="$4"
  local batch_size="$5"

  if [[ -e "$out_dir" ]]; then
    echo "[ERROR] Rerun target already exists: $out_dir" >&2
    echo "        Remove only this target dir if you intentionally want to rerun it." >&2
    return 1
  fi
  mkdir -p "$out_dir"

  cmd=(
    "$PYTHON" -u scripts/train_mathspeech_projector_best_wer.py
    --train_csv "${TRAIN_CSV}"
    --valid_csv "${VAL_CSV}"
    --test_csv "${TEST_CSV}"
    --save_dir "$out_dir"
    --clap_emb_path "${CLAP_EMB}"
    --whisper_name "${MODEL}"
    --adapter_type gated
    --pool_type cls
    --adapter_bottleneck 256
    --dropout 0.1
    --adapter_scale_init 0.01
    --align_loss_type cosine
    --lambda_align 0.05
    --lambda_hidden 0.1
    --batch_size "$batch_size"
    --epochs "${EPOCHS}"
    --lr "$lr"
    --weight_decay 1e-4
    --selection_num_beams "${SELECTION_BEAM}"
    --selection_max_new_tokens "${MAX_NEW_TOKENS}"
    --seed "${SEED}"
    --num_workers "${NUM_WORKERS}"
  )

  if [[ "$freeze" == "true" ]]; then
    cmd+=(--freeze_whisper)
    echo "[TRAIN] Ours | gpu=${gpu} lr=${lr} | Whisper frozen"
  else
    echo "[TRAIN] CLAP-guided Full FT | gpu=${gpu} lr=${lr} | Whisper trainable"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 | tee "${out_dir}/train.log"

  for beam in "${BEAMS[@]}"; do
    run_eval_projector "$gpu" "${out_dir}/best.pt" "$out_dir" "$beam"
  done
}

pids=()
names=()
launch(){
  local name="$1"; shift
  echo "[LAUNCH] $name"
  ( "$@" ) > "${OUT_ROOT}/logs/${name}.launcher.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

launch "ours_lr1e-5_gpu${GPU_LIST[0]}" \
  run_projector_job "${GPU_LIST[0]}" 1e-5 true \
  "${OUT_ROOT}/ours_lr1e-5" "${OURS_BATCH_SIZE}"

launch "ours_lr1e-4_gpu${GPU_LIST[1]}" \
  run_projector_job "${GPU_LIST[1]}" 1e-4 true \
  "${OUT_ROOT}/ours_lr1e-4" "${OURS_BATCH_SIZE}"

launch "clap_fullft_lr1e-5_gpu${GPU_LIST[2]}" \
  run_projector_job "${GPU_LIST[2]}" 1e-5 false \
  "${OUT_ROOT}/clap_fullft_lr1e-5" "${CLAP_FULLFT_BATCH_SIZE}"

launch "clap_fullft_lr1e-4_gpu${GPU_LIST[3]}" \
  run_projector_job "${GPU_LIST[3]}" 1e-4 false \
  "${OUT_ROOT}/clap_fullft_lr1e-4" "${CLAP_FULLFT_BATCH_SIZE}"

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[DONE] ${names[$i]}"
  else
    echo "[FAILED] ${names[$i]}" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[ERROR] One or more projector reruns failed. Check ${OUT_ROOT}/logs/" >&2
  exit "$status"
fi

"${PYTHON}" -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}"

echo "============================================================"
echo "[DONE] Unified best-valid-WER results"
echo "${OUT_ROOT}/common_asr_results.md"
echo "${OUT_ROOT}/common_asr_results.csv"
echo "============================================================"
