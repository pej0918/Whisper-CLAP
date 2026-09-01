#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Speech2LaTeX -> spoken-math ASR baseline runner
# ============================================================
#
# The manifests are produced by:
#   scripts/prepare_s2l_asr_from_arrow.py
#
# Main protocol:
#   train = Mix
#   valid = Mix
#   test  = Mix / H / A
#
# Source-type ablation can use exactly the same runner:
#   train = H or A
#   valid = same source as train by default
#   test  = Mix / H / A
#
# Examples:
#
#   # Main S2L-Sent experiment
#   bash command/run_s2l_common_baselines.sh \
#     --task sent \
#     --train_source mix
#
#   # Main S2L-Eq experiment
#   bash command/run_s2l_common_baselines.sh \
#     --task eq \
#     --train_source mix
#
#   # Human-only training ablation
#   bash command/run_s2l_common_baselines.sh \
#     --task eq \
#     --train_source h
#
#   # Artificial-only training, Human test only
#   bash command/run_s2l_common_baselines.sh \
#     --task eq \
#     --train_source a \
#     --test_sources h
#
#   # Explicit all test subsets
#   bash command/run_s2l_common_baselines.sh \
#     --task sent \
#     --train_source mix \
#     --test_sources mix,h,a
#
# Environment overrides such as GPU=0, EPOCHS=10, NUM_BEAMS=5,
# OUT_ROOT=..., etc. remain supported.

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------

GPU="${GPU:-0}"
ROOT="${ROOT:-$PWD}"
S2L_ROOT="${S2L_ROOT:-/data1/eunju/datasets/speech2latex_asr_seed42}"
TASK="${TASK:-sent}"
TRAIN_SOURCE="${TRAIN_SOURCE:-mix}"
VALID_SOURCE="${VALID_SOURCE:-auto}"
TEST_SOURCES="${TEST_SOURCES:-mix,h,a}"

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

# If OUT_ROOT is not explicitly supplied, keep each source protocol isolated.
OUT_ROOT_USER_SET=0
if [[ -n "${OUT_ROOT+x}" ]]; then
  OUT_ROOT_USER_SET=1
fi

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  bash command/run_s2l_common_baselines.sh [options]

Options:
  --task {sent|eq}
      S2L-Sentences or S2L-Equations.

  --train_source {mix|h|a}
      Training source type. Default: mix.

  --valid_source {auto|mix|h|a}
      Validation source type. Default: auto, i.e. same as train_source.

  --test_sources <comma-separated list>
      Test source types. Default: mix,h,a.
      Examples: mix,h,a   h   h,a

  --s2l_root PATH
      Root produced by prepare_s2l_asr_from_arrow.py.

  --out_root PATH
      Output root. If omitted:
      /data1/eunju/clap_whisper_results/s2l_asr/<task>/train_<source>

  --gpu ID
      CUDA device id. Default: 0.

  --num_beams N
      Evaluation beam size and validation-selection beam size. Default: 5.

  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --train_source)
      TRAIN_SOURCE="$2"
      shift 2
      ;;
    --valid_source)
      VALID_SOURCE="$2"
      shift 2
      ;;
    --test_sources)
      TEST_SOURCES="$2"
      shift 2
      ;;
    --s2l_root)
      S2L_ROOT="$2"
      shift 2
      ;;
    --out_root)
      OUT_ROOT="$2"
      OUT_ROOT_USER_SET=1
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --num_beams)
      NUM_BEAMS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

# ------------------------------------------------------------
# Validate protocol arguments
# ------------------------------------------------------------

case "${TASK}" in
  sent)
    TASK_DIR="${S2L_ROOT}/s2l_sent"
    PREFIX="s2l_sent"
    ;;
  eq)
    TASK_DIR="${S2L_ROOT}/s2l_eq"
    PREFIX="s2l_eq"
    ;;
  *)
    echo "[ERROR] --task must be sent or eq; got: ${TASK}"
    exit 2
    ;;
esac

validate_source() {
  local src="$1"
  case "${src}" in
    mix|h|a) ;;
    *)
      echo "[ERROR] source must be one of mix, h, a; got: ${src}"
      exit 2
      ;;
  esac
}

validate_source "${TRAIN_SOURCE}"

if [[ "${VALID_SOURCE}" == "auto" ]]; then
  VALID_SOURCE="${TRAIN_SOURCE}"
else
  validate_source "${VALID_SOURCE}"
fi

# Accept both "mix,h,a" and a single source such as "h".
IFS=',' read -r -a TEST_SOURCE_ARRAY <<< "${TEST_SOURCES}"
if [[ ${#TEST_SOURCE_ARRAY[@]} -eq 0 ]]; then
  echo "[ERROR] --test_sources cannot be empty"
  exit 2
fi
for src in "${TEST_SOURCE_ARRAY[@]}"; do
  validate_source "${src}"
done

TRAIN_CSV="${TASK_DIR}/${PREFIX}_train_${TRAIN_SOURCE}.csv"
VAL_CSV="${TASK_DIR}/${PREFIX}_valid_${VALID_SOURCE}.csv"

if [[ ${OUT_ROOT_USER_SET} -eq 0 ]]; then
  OUT_ROOT="/data1/eunju/clap_whisper_results/s2l_asr/${TASK}/train_${TRAIN_SOURCE}"
fi

# ------------------------------------------------------------
# Check manifests
# ------------------------------------------------------------

for f in "${TRAIN_CSV}" "${VAL_CSV}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing manifest: ${f}"
    echo "Run scripts/prepare_s2l_asr_from_arrow.py first."
    exit 1
  fi
done

for src in "${TEST_SOURCE_ARRAY[@]}"; do
  f="${TASK_DIR}/${PREFIX}_test_${src}.csv"
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing test manifest: ${f}"
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" \
  "${OUT_ROOT}/whisper_base" \
  "${OUT_ROOT}/lora_fair_lr1e-5" \
  "${OUT_ROOT}/lora_paper_lr1e-4" \
  "${OUT_ROOT}/residual_b256"

printf '\n[CONFIG]\n'
echo "TASK          : ${TASK}"
echo "S2L_ROOT      : ${S2L_ROOT}"
echo "TASK_DIR      : ${TASK_DIR}"
echo "TRAIN_SOURCE  : ${TRAIN_SOURCE}"
echo "VALID_SOURCE  : ${VALID_SOURCE}"
echo "TEST_SOURCES  : ${TEST_SOURCES}"
echo "TRAIN_CSV     : ${TRAIN_CSV}"
echo "VAL_CSV       : ${VAL_CSV}"
echo "OUT_ROOT      : ${OUT_ROOT}"
echo "GPU           : ${GPU}"
echo "MODEL         : ${MODEL}"
echo "EPOCHS        : ${EPOCHS}"
echo "BEAM          : ${NUM_BEAMS}"

# Save protocol metadata beside results for reproducibility.
cat > "${OUT_ROOT}/protocol.txt" <<EOF
TASK=${TASK}
TRAIN_SOURCE=${TRAIN_SOURCE}
VALID_SOURCE=${VALID_SOURCE}
TEST_SOURCES=${TEST_SOURCES}
TRAIN_CSV=${TRAIN_CSV}
VAL_CSV=${VAL_CSV}
MODEL=${MODEL}
SEED=${SEED}
EPOCHS=${EPOCHS}
NUM_BEAMS=${NUM_BEAMS}
EOF

# ------------------------------------------------------------
# Shared evaluation helpers
# ------------------------------------------------------------

eval_whisper_sources() {
  local model_path="$1"
  local method_dir="$2"

  for src in "${TEST_SOURCE_ARRAY[@]}"; do
    local test_csv="${TASK_DIR}/${PREFIX}_test_${src}.csv"
    local eval_dir="${method_dir}/test_${src}"
    mkdir -p "${eval_dir}"

    echo
    echo "[EVAL] model=${model_path} test_source=${src} beam=${NUM_BEAMS}"

    CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval_whisper.py \
      --manifest "${test_csv}" \
      --model "${model_path}" \
      --output_csv "${eval_dir}/predictions_beam${NUM_BEAMS}.csv" \
      --summary_json "${eval_dir}/summary_beam${NUM_BEAMS}.json" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --num_beams "${NUM_BEAMS}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      2>&1 | tee "${eval_dir}/eval_beam${NUM_BEAMS}.log"
  done
}

eval_residual_sources() {
  local ckpt="$1"
  local method_dir="$2"

  for src in "${TEST_SOURCE_ARRAY[@]}"; do
    local test_csv="${TASK_DIR}/${PREFIX}_test_${src}.csv"
    local eval_dir="${method_dir}/test_${src}"
    mkdir -p "${eval_dir}"

    echo
    echo "[EVAL Residual] test_source=${src} beam=${NUM_BEAMS}"

    CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval_whisper_residual_adapter.py \
      --manifest "${test_csv}" \
      --ckpt "${ckpt}" \
      --output_csv "${eval_dir}/predictions_beam${NUM_BEAMS}.csv" \
      --summary_json "${eval_dir}/summary_beam${NUM_BEAMS}.json" \
      --model "${MODEL}" \
      --adapter_bottleneck 256 \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --num_beams "${NUM_BEAMS}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      2>&1 | tee "${eval_dir}/eval_beam${NUM_BEAMS}.log"
  done
}

# ------------------------------------------------------------
# 0) Frozen Whisper-base
# ------------------------------------------------------------

eval_whisper_sources \
  "${MODEL}" \
  "${OUT_ROOT}/whisper_base"

# ------------------------------------------------------------
# 1) LoRA, controlled LR=1e-5
# ------------------------------------------------------------

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
  --num_workers "${NUM_WORKERS}" \
  --generation_num_beams "${NUM_BEAMS}" \
  --generation_max_length "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_ROOT}/lora_fair_lr1e-5/train.log"

eval_whisper_sources \
  "${OUT_ROOT}/lora_fair_lr1e-5/merged" \
  "${OUT_ROOT}/lora_fair_lr1e-5"

# ------------------------------------------------------------
# 2) LoRA, paper LR=1e-4
# ------------------------------------------------------------

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
  --num_workers "${NUM_WORKERS}" \
  --generation_num_beams "${NUM_BEAMS}" \
  --generation_max_length "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_ROOT}/lora_paper_lr1e-4/train.log"

eval_whisper_sources \
  "${OUT_ROOT}/lora_paper_lr1e-4/merged" \
  "${OUT_ROOT}/lora_paper_lr1e-4"

# ------------------------------------------------------------
# 3) KAUST-style residual adapter
# ------------------------------------------------------------

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

eval_residual_sources \
  "${OUT_ROOT}/residual_b256/best.pt" \
  "${OUT_ROOT}/residual_b256"

# ------------------------------------------------------------
# Collect what the existing collector understands.
# H/A/Mix summaries are additionally kept in each test_* dir.
# ------------------------------------------------------------

python -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}" || true

echo
echo "[DONE] ${TASK}: train=${TRAIN_SOURCE}, valid=${VALID_SOURCE}, tests=${TEST_SOURCES}"
echo "Results: ${OUT_ROOT}"
