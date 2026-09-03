#!/usr/bin/env bash
set -euo pipefail

# Final MathSpeech protocol:
#   LR ∈ {1e-5, 1e-4, 3e-4}
#   10 epochs per LR
#   checkpoint selection = best validation WER @ beam 5
#   LR selection         = best validation WER across the 3 LR runs
#   test                 = selected LR only, beam 5
#
# Examples:
#   METHOD=ours GPUS="1 2 3" bash command/run_mathspeech_final_protocol.sh
#   METHOD=lora_noout GPUS="1 2 3" bash command/run_mathspeech_final_protocol.sh
#
# Optional:
#   TRAIN_ONLY=1 ...   # train all LRs + select, but do not test

ROOT="${ROOT:-$PWD}"
METHOD="${METHOD:-ours}"
SEED="${SEED:-42}"
LRS_STR="${LRS:-1e-5 1e-4 3e-4}"
GPUS_STR="${GPUS:-0 0 0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_final_protocol_seed${SEED}}"

read -r -a LRS <<< "${LRS_STR}"
read -r -a GPUS <<< "${GPUS_STR}"

if [[ ${#LRS[@]} -ne 3 ]]; then
  echo "[ERROR] Final protocol requires exactly 3 LRs: 1e-5 1e-4 3e-4" >&2
  exit 2
fi
if [[ ${#GPUS[@]} -ne 3 ]]; then
  echo "[ERROR] Provide exactly 3 GPU ids, e.g. GPUS=\"1 2 3\"" >&2
  exit 2
fi

case "${METHOD}" in
  fullft|clap_fullft|lora_noout|lora_out|residual_b256|ours|rq5b_lora_r7|rq5b_residual_b128) ;;
  whisper_base)
    GPU="${GPUS[0]}" OUT_ROOT="${OUT_ROOT}" METHOD=whisper_base STAGE=test \
      bash command/run_mathspeech_reported_configs.sh
    exit 0
    ;;
  *)
    echo "[ERROR] METHOD=${METHOD} is not an LR-search method in the final protocol." >&2
    echo "Use run_mathspeech_reported_configs.sh directly for fixed-LR RQ4 ablations." >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_ROOT}/logs"

echo "================================================================================"
echo "MATHSPEECH FINAL LR SEARCH"
echo "METHOD=${METHOD}"
echo "LRS=${LRS[*]}"
echo "GPUS=${GPUS[*]}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "================================================================================"

pids=()
for i in 0 1 2; do
  lr="${LRS[$i]}"
  gpu="${GPUS[$i]}"
  log="${OUT_ROOT}/logs/${METHOD}_lr${lr}.train.log"
  echo "[TRAIN] METHOD=${METHOD} LR=${lr} GPU=${gpu}"
  (
    METHOD="${METHOD}" LR="${lr}" GPU="${gpu}" OUT_ROOT="${OUT_ROOT}" STAGE=train \
      bash command/run_mathspeech_reported_configs.sh
  ) 2>&1 | tee "${log}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] At least one LR training run failed." >&2
  exit 1
fi

python scripts/select_best_lr.py \
  --root "${OUT_ROOT}" \
  --method "${METHOD}" \
  --lrs "${LRS[@]}" \
  --output "${OUT_ROOT}/${METHOD}_lr_selection.json"

SELECTED_LR=$(python - <<PY
import json
print(json.load(open("${OUT_ROOT}/${METHOD}_lr_selection.json"))["selected_lr"])
PY
)

echo "[SELECTED] METHOD=${METHOD} LR=${SELECTED_LR}"

if [[ "${TRAIN_ONLY}" == "1" ]]; then
  echo "TRAIN_ONLY=1: skipping test evaluation"
  exit 0
fi

# Reuse one available GPU for the single final test evaluation.
METHOD="${METHOD}" LR="${SELECTED_LR}" GPU="${GPUS[0]}" OUT_ROOT="${OUT_ROOT}" STAGE=test \
  bash command/run_mathspeech_reported_configs.sh \
  2>&1 | tee "${OUT_ROOT}/logs/${METHOD}_selected_lr${SELECTED_LR}.test.log"

echo "DONE: METHOD=${METHOD} selected_lr=${SELECTED_LR}"
