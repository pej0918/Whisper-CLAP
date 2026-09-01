#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Speech2LaTeX -> spoken-math ASR: full controlled 7-method grid
# ============================================================================
# Main protocol (default):
#   Train = Mix
#   Valid = Mix
#   Test  = Mix / H / A
#   Epoch = 10
#   LR    = {1e-5, 1e-4}
#   checkpoint selection = lowest validation WER with beam=5
#   test decoding        = beam {1, 5}
#
# Methods:
#   0. Whisper-base (no training)
#   1. Full Fine-tuning
#   2. CLAP-guided Full Fine-tuning
#   3. LoRA +out_proj
#   4. LoRA -out_proj
#   5. KAUST-style Residual Adapter
#   6. Ours (frozen Whisper + gated CLAP adapter/projector)
#
# Examples:
#   GPUS="0 1 2 3 4 5" bash command/run_s2l_common_grid_lr_beam_parallel_6gpu.sh --task sent
#   GPUS="0 1 2 3 4 5" bash command/run_s2l_common_grid_lr_beam_parallel_6gpu.sh --task eq
#
# Optional source ablation later:
#   ... --task eq --train_source h
#   ... --task sent --train_source a --test_sources h,a
#
# NOTE:
#   H/A/Mix all use the same task-level global CLAP embedding tensor.
#   sample_id is global and CLAP lookup is embedding[sample_id - 1].

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"

S2L_ROOT="${S2L_ROOT:-/data1/eunju/datasets/speech2latex_asr_seed42}"
TASK="${TASK:-sent}"
TRAIN_SOURCE="${TRAIN_SOURCE:-mix}"
VALID_SOURCE="${VALID_SOURCE:-auto}"
TEST_SOURCES="${TEST_SOURCES:-mix,h,a}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
OURS_BATCH_SIZE="${OURS_BATCH_SIZE:-4}"
CLAP_FULLFT_BATCH_SIZE="${CLAP_FULLFT_BATCH_SIZE:-4}"
RESIDUAL_BATCH_SIZE="${RESIDUAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SELECTION_BEAM="${SELECTION_BEAM:-5}"

GPUS_STR="${GPUS:-0 1 2 3 4 5}"
read -r -a GPU_LIST <<< "${GPUS_STR}"
if [[ ${#GPU_LIST[@]} -lt 6 ]]; then
  echo "[ERROR] Need 6 GPU ids in GPUS, e.g. GPUS=\"0 1 2 3 4 5\"" >&2
  exit 2
fi

LRS_STR="${LRS:-1e-5 1e-4}"
read -r -a LRS <<< "${LRS_STR}"
if [[ ${#LRS[@]} -ne 2 ]]; then
  echo "[ERROR] This controlled runner expects exactly two LRs; got: ${LRS[*]}" >&2
  exit 2
fi

BEAMS_STR="${BEAMS:-1 5}"
read -r -a BEAMS <<< "${BEAMS_STR}"

LORA_TARGETS_WITH_OUTPROJ="${LORA_TARGETS_WITH_OUTPROJ:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"
LORA_TARGETS_NO_OUTPROJ="${LORA_TARGETS_NO_OUTPROJ:-q_proj,k_proj,v_proj,fc1,fc2}"

OUT_ROOT_USER_SET=0
if [[ -n "${OUT_ROOT+x}" ]]; then
  OUT_ROOT_USER_SET=1
fi

usage(){
  cat <<'EOF'
Usage:
  bash command/run_s2l_common_grid_lr_beam_parallel_6gpu.sh [options]

Options:
  --task {sent|eq}
  --train_source {mix|h|a}       default: mix
  --valid_source {auto|mix|h|a}  default: auto (= train_source)
  --test_sources LIST            default: mix,h,a
  --s2l_root PATH
  --out_root PATH
  --help

Environment:
  GPUS="0 1 2 3 4 5"
  EPOCHS=10
  LRS="1e-5 1e-4"
  BEAMS="1 5"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --train_source) TRAIN_SOURCE="$2"; shift 2 ;;
    --valid_source) VALID_SOURCE="$2"; shift 2 ;;
    --test_sources) TEST_SOURCES="$2"; shift 2 ;;
    --s2l_root) S2L_ROOT="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; OUT_ROOT_USER_SET=1; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

validate_source(){
  case "$1" in mix|h|a) ;; *) echo "[ERROR] source must be mix/h/a; got $1" >&2; exit 2 ;; esac
}
validate_source "${TRAIN_SOURCE}"
if [[ "${VALID_SOURCE}" == "auto" ]]; then
  VALID_SOURCE="${TRAIN_SOURCE}"
else
  validate_source "${VALID_SOURCE}"
fi
IFS=',' read -r -a TEST_SOURCE_ARRAY <<< "${TEST_SOURCES}"
for src in "${TEST_SOURCE_ARRAY[@]}"; do validate_source "${src}"; done

case "${TASK}" in
  sent)
    TASK_DIR="${S2L_ROOT}/s2l_sent"
    PREFIX="s2l_sent"
    CLAP_EMB_DEFAULT="${TASK_DIR}/s2l_sent_clap_text_emb.pt"
    ;;
  eq)
    TASK_DIR="${S2L_ROOT}/s2l_eq"
    PREFIX="s2l_eq"
    CLAP_EMB_DEFAULT="${TASK_DIR}/s2l_eq_clap_text_emb.pt"
    ;;
  *) echo "[ERROR] --task must be sent or eq; got ${TASK}" >&2; exit 2 ;;
esac

TRAIN_CSV="${TASK_DIR}/${PREFIX}_train_${TRAIN_SOURCE}.csv"
VAL_CSV="${TASK_DIR}/${PREFIX}_valid_${VALID_SOURCE}.csv"
# Projector trainer only uses this for split-integrity bookkeeping; model selection
# still uses VAL_CSV. Keep the canonical Mix test split here.
CHECK_TEST_CSV="${TASK_DIR}/${PREFIX}_test_mix.csv"
CLAP_EMB="${CLAP_EMB:-${CLAP_EMB_DEFAULT}}"

if [[ ${OUT_ROOT_USER_SET} -eq 0 ]]; then
  OUT_ROOT="/data1/eunju/clap_whisper_results/s2l_common_grid_bestvalidwer/${TASK}/train_${TRAIN_SOURCE}_seed${SEED}_6gpu"
fi
mkdir -p "${OUT_ROOT}/logs"

require_file(){ [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
require_file "${TRAIN_CSV}"
require_file "${VAL_CSV}"
require_file "${CHECK_TEST_CSV}"
require_file "${CLAP_EMB}"
for src in "${TEST_SOURCE_ARRAY[@]}"; do
  require_file "${TASK_DIR}/${PREFIX}_test_${src}.csv"
done

cat > "${OUT_ROOT}/protocol.txt" <<EOF
TASK=${TASK}
TRAIN_SOURCE=${TRAIN_SOURCE}
VALID_SOURCE=${VALID_SOURCE}
TEST_SOURCES=${TEST_SOURCES}
TRAIN_CSV=${TRAIN_CSV}
VAL_CSV=${VAL_CSV}
CHECK_TEST_CSV=${CHECK_TEST_CSV}
CLAP_EMB=${CLAP_EMB}
MODEL=${MODEL}
SEED=${SEED}
EPOCHS=${EPOCHS}
LRS=${LRS[*]}
SELECTION_BEAM=${SELECTION_BEAM}
TEST_BEAMS=${BEAMS[*]}
EOF

echo "================================================================================"
echo "S2L CONTROLLED COMMON GRID"
echo "================================================================================"
echo "TASK             : ${TASK}"
echo "TRAIN_SOURCE     : ${TRAIN_SOURCE}"
echo "VALID_SOURCE     : ${VALID_SOURCE}"
echo "TEST_SOURCES     : ${TEST_SOURCE_ARRAY[*]}"
echo "TRAIN_CSV        : ${TRAIN_CSV}"
echo "VAL_CSV          : ${VAL_CSV}"
echo "CLAP_EMB         : ${CLAP_EMB}"
echo "OUT_ROOT         : ${OUT_ROOT}"
echo "GPUS             : ${GPU_LIST[*]}"
echo "EPOCHS           : ${EPOCHS}"
echo "LRS              : ${LRS[*]}"
echo "SELECTION_BEAM   : ${SELECTION_BEAM}"
echo "TEST_BEAMS       : ${BEAMS[*]}"
echo "================================================================================"

# ---------------------------------------------------------------------------
# Evaluation helpers: same checkpoint -> every requested source x beam.
# ---------------------------------------------------------------------------
run_eval_whisper_sources(){
  local gpu="$1" model_path="$2" method_dir="$3"
  for src in "${TEST_SOURCE_ARRAY[@]}"; do
    local manifest="${TASK_DIR}/${PREFIX}_test_${src}.csv"
    local eval_dir="${method_dir}/test_${src}"
    mkdir -p "${eval_dir}"
    for beam in "${BEAMS[@]}"; do
      echo "[EVAL Whisper] gpu=${gpu} source=${src} beam=${beam} model=${model_path}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_whisper.py \
        --manifest "${manifest}" \
        --model "${model_path}" \
        --output_csv "${eval_dir}/predictions_beam${beam}.csv" \
        --summary_json "${eval_dir}/summary_beam${beam}.json" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --num_beams "${beam}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        2>&1 | tee "${eval_dir}/eval_beam${beam}.log"
    done
  done
}

run_eval_projector_sources(){
  local gpu="$1" ckpt="$2" method_dir="$3"
  for src in "${TEST_SOURCE_ARRAY[@]}"; do
    local manifest="${TASK_DIR}/${PREFIX}_test_${src}.csv"
    local eval_dir="${method_dir}/test_${src}"
    mkdir -p "${eval_dir}"
    for beam in "${BEAMS[@]}"; do
      echo "[EVAL Projector] gpu=${gpu} source=${src} beam=${beam} ckpt=${ckpt}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_mathspeech_projector_source_disjoint_hf.py \
        --manifest "${manifest}" \
        --ckpt "${ckpt}" \
        --output_csv "${eval_dir}/predictions_beam${beam}.csv" \
        --summary_json "${eval_dir}/summary_beam${beam}.json" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --num_beams "${beam}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --fp16 \
        2>&1 | tee "${eval_dir}/eval_beam${beam}.log"
    done
  done
}

run_eval_residual_sources(){
  local gpu="$1" ckpt="$2" method_dir="$3"
  for src in "${TEST_SOURCE_ARRAY[@]}"; do
    local manifest="${TASK_DIR}/${PREFIX}_test_${src}.csv"
    local eval_dir="${method_dir}/test_${src}"
    mkdir -p "${eval_dir}"
    for beam in "${BEAMS[@]}"; do
      echo "[EVAL Residual] gpu=${gpu} source=${src} beam=${beam} ckpt=${ckpt}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
        --manifest "${manifest}" \
        --ckpt "${ckpt}" \
        --model "${MODEL}" \
        --adapter_bottleneck 256 \
        --output_csv "${eval_dir}/predictions_beam${beam}.csv" \
        --summary_json "${eval_dir}/summary_beam${beam}.json" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --num_beams "${beam}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        2>&1 | tee "${eval_dir}/eval_beam${beam}.log"
    done
  done
}

# ---------------------------------------------------------------------------
# Train jobs
# ---------------------------------------------------------------------------
run_fullft_job(){
  local gpu="$1" lr="$2"
  local out_dir="${OUT_ROOT}/fullft_lr${lr}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_fullft_compat.py \
    --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${out_dir}" --model "${MODEL}" \
    --epochs "${EPOCHS}" --learning_rate "${lr}" \
    --train_batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
    --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" 2>&1 | tee "${out_dir}/train.log"
  run_eval_whisper_sources "${gpu}" "${out_dir}/best" "${out_dir}"
}

run_projector_job(){
  local gpu="$1" lr="$2" freeze_flag="$3" out_dir="$4"
  mkdir -p "${out_dir}"
  local freeze_args=()
  local bs="${CLAP_FULLFT_BATCH_SIZE}"
  if [[ "${freeze_flag}" == "true" ]]; then
    freeze_args+=(--freeze_whisper)
    bs="${OURS_BATCH_SIZE}"
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_mathspeech_projector_best_wer.py \
    --train_csv "${TRAIN_CSV}" --valid_csv "${VAL_CSV}" --test_csv "${CHECK_TEST_CSV}" \
    --save_dir "${out_dir}" --clap_emb_path "${CLAP_EMB}" --whisper_name "${MODEL}" \
    --adapter_type gated --pool_type cls --adapter_bottleneck 256 \
    --dropout 0.1 --adapter_scale_init 0.01 \
    --align_loss_type cosine --lambda_align 0.05 --lambda_hidden 0.1 \
    --batch_size "${bs}" --epochs "${EPOCHS}" --lr "${lr}" --weight_decay 1e-4 \
    --selection_num_beams "${SELECTION_BEAM}" --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" --num_workers "${NUM_WORKERS}" "${freeze_args[@]}" \
    2>&1 | tee "${out_dir}/train.log"

  run_eval_projector_sources "${gpu}" "${out_dir}/best.pt" "${out_dir}"
}

run_ours_job(){
  local gpu="$1" lr="$2"
  run_projector_job "${gpu}" "${lr}" true "${OUT_ROOT}/ours_lr${lr}"
}

run_clap_fullft_job(){
  local gpu="$1" lr="$2"
  run_projector_job "${gpu}" "${lr}" false "${OUT_ROOT}/clap_fullft_lr${lr}"
}

run_lora_job(){
  local gpu="$1" lr="$2" variant="$3" targets="$4"
  local out_dir="${OUT_ROOT}/lora_whisper_lr${lr}_${variant}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_lora_controlled.py \
    --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${out_dir}" --model "${MODEL}" \
    --rank 32 --lora_alpha 32 --target_modules "${targets}" \
    --epochs "${EPOCHS}" --learning_rate "${lr}" \
    --train_batch_size "${BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
    --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" 2>&1 | tee "${out_dir}/train.log"
  run_eval_whisper_sources "${gpu}" "${out_dir}/merged" "${out_dir}"
}

run_residual_job(){
  local gpu="$1" lr="$2"
  local out_dir="${OUT_ROOT}/residual_b256_lr${lr}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
    --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --save_dir "${out_dir}" --model "${MODEL}" \
    --epochs "${EPOCHS}" --lr "${lr}" --batch_size "${RESIDUAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" --adapter_bottleneck 256 \
    --selection_num_beams "${SELECTION_BEAM}" --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" --fp16 2>&1 | tee "${out_dir}/train.log"
  run_eval_residual_sources "${gpu}" "${out_dir}/best.pt" "${out_dir}"
}

# ---------------------------------------------------------------------------
# 0) Frozen Whisper-base: one model, all source/beam evaluations.
# ---------------------------------------------------------------------------
mkdir -p "${OUT_ROOT}/whisper_base"
run_eval_whisper_sources "${GPU_LIST[0]}" "${MODEL}" "${OUT_ROOT}/whisper_base"

# ---------------------------------------------------------------------------
# Parallel launcher. 12 train jobs = 2 waves x 6 GPUs.
# ---------------------------------------------------------------------------
pids=()
names=()
launch(){
  local name="$1"; shift
  echo "[LAUNCH] ${name}"
  ( "$@" ) > "${OUT_ROOT}/logs/${name}.launcher.log" 2>&1 &
  pids+=("$!")
  names+=("${name}")
}
wait_wave(){
  local status=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[DONE] ${names[$i]}"
    else
      echo "[FAILED] ${names[$i]}" >&2
      status=1
    fi
  done
  pids=(); names=()
  [[ ${status} -eq 0 ]] || { echo "[ERROR] Wave failed" >&2; exit ${status}; }
}

LR1="${LRS[0]}"
LR2="${LRS[1]}"

# Wave 1: FullFT, Ours, LoRA +out_proj
launch "fullft_lr${LR1}_gpu${GPU_LIST[0]}" run_fullft_job "${GPU_LIST[0]}" "${LR1}"
launch "fullft_lr${LR2}_gpu${GPU_LIST[1]}" run_fullft_job "${GPU_LIST[1]}" "${LR2}"
launch "ours_lr${LR1}_gpu${GPU_LIST[2]}" run_ours_job "${GPU_LIST[2]}" "${LR1}"
launch "ours_lr${LR2}_gpu${GPU_LIST[3]}" run_ours_job "${GPU_LIST[3]}" "${LR2}"
launch "lora_outproj_lr${LR1}_gpu${GPU_LIST[4]}" run_lora_job "${GPU_LIST[4]}" "${LR1}" outproj "${LORA_TARGETS_WITH_OUTPROJ}"
launch "lora_outproj_lr${LR2}_gpu${GPU_LIST[5]}" run_lora_job "${GPU_LIST[5]}" "${LR2}" outproj "${LORA_TARGETS_WITH_OUTPROJ}"
wait_wave

# Wave 2: Residual, LoRA -out_proj, CLAP-guided FullFT
launch "residual_lr${LR1}_gpu${GPU_LIST[0]}" run_residual_job "${GPU_LIST[0]}" "${LR1}"
launch "residual_lr${LR2}_gpu${GPU_LIST[1]}" run_residual_job "${GPU_LIST[1]}" "${LR2}"
launch "lora_nooutproj_lr${LR1}_gpu${GPU_LIST[2]}" run_lora_job "${GPU_LIST[2]}" "${LR1}" nooutproj "${LORA_TARGETS_NO_OUTPROJ}"
launch "lora_nooutproj_lr${LR2}_gpu${GPU_LIST[3]}" run_lora_job "${GPU_LIST[3]}" "${LR2}" nooutproj "${LORA_TARGETS_NO_OUTPROJ}"
launch "clap_fullft_lr${LR1}_gpu${GPU_LIST[4]}" run_clap_fullft_job "${GPU_LIST[4]}" "${LR1}"
launch "clap_fullft_lr${LR2}_gpu${GPU_LIST[5]}" run_clap_fullft_job "${GPU_LIST[5]}" "${LR2}"
wait_wave

"${PYTHON}" -u scripts/collect_s2l_asr_results.py --output_dir "${OUT_ROOT}"

echo "================================================================================"
echo "DONE"
echo "Task    : ${TASK}"
echo "Results : ${OUT_ROOT}"
echo "Full grid table       : ${OUT_ROOT}/s2l_asr_results.csv"
echo "Validation-selected LR: ${OUT_ROOT}/s2l_best_valid_lr_results.csv"
echo "================================================================================"
