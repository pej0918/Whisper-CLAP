#!/usr/bin/env bash
set -euo pipefail

# Final Human-only S2L protocol:
#   filters              = language=eng, is_tts=0, duration<=30 sec
#   LR                   = {1e-5, 1e-4, 3e-4}
#   max epochs           = 10
#   checkpoint selection = best validation WER @ beam 5 within each LR
#   LR selection         = best validation WER across the 3 LR runs
#   test                 = selected LR only @ beam 5
#
# Examples:
#   TASK=sent METHOD=ours GPUS="1 2 3" bash command/run_s2l_human_final_protocol.sh
#   TASK=eq METHOD=lora_noout GPUS="1 2 3" bash command/run_s2l_human_final_protocol.sh

TASK="${TASK:-sent}"
METHOD="${METHOD:-ours}"
SEED="${SEED:-42}"
LRS_STR="${LRS:-1e-5 1e-4 3e-4}"
GPUS_STR="${GPUS:-0 0 0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/s2l_human_final/${TASK}_seed${SEED}}"

read -r -a LRS <<< "${LRS_STR}"
read -r -a GPUS <<< "${GPUS_STR}"
[[ ${#LRS[@]} -eq 3 ]] || { echo "[ERROR] Need exactly 3 LRs" >&2; exit 2; }
[[ ${#GPUS[@]} -eq 3 ]] || { echo "[ERROR] Need exactly 3 GPU ids" >&2; exit 2; }

if [[ "${METHOD}" == "whisper_base" ]]; then
  TASK="${TASK}" METHOD=whisper_base GPU="${GPUS[0]}" OUT_ROOT="${OUT_ROOT}" STAGE=test \
    bash command/run_s2l_human_config.sh
  exit 0
fi

case "${METHOD}" in
  fullft|clap_fullft|lora_noout|lora_out|residual_b256|ours|rq5b_lora_r7|rq5b_residual_b128) ;;
  *)
    echo "[ERROR] METHOD=${METHOD} is not an LR-search method. Use run_s2l_human_config.sh directly for fixed-LR RQ4." >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_ROOT}/logs"
pids=()
for i in 0 1 2; do
  lr="${LRS[$i]}"; gpu="${GPUS[$i]}"
  echo "[TRAIN] TASK=${TASK} METHOD=${METHOD} LR=${lr} GPU=${gpu}"
  (
    TASK="${TASK}" METHOD="${METHOD}" LR="${lr}" GPU="${gpu}" OUT_ROOT="${OUT_ROOT}" STAGE=train \
      bash command/run_s2l_human_config.sh
  ) 2>&1 | tee "${OUT_ROOT}/logs/${METHOD}_lr${lr}.train.log" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
[[ "${failed}" -eq 0 ]] || { echo "[ERROR] One or more LR runs failed" >&2; exit 1; }

python scripts/select_best_lr.py \
  --root "${OUT_ROOT}" --method "${METHOD}" --lrs "${LRS[@]}" \
  --output "${OUT_ROOT}/${METHOD}_lr_selection.json"

SELECTED_LR=$(python - <<PY
import json
print(json.load(open("${OUT_ROOT}/${METHOD}_lr_selection.json"))["selected_lr"])
PY
)
echo "[SELECTED] TASK=${TASK} METHOD=${METHOD} LR=${SELECTED_LR}"

if [[ "${TRAIN_ONLY}" == "1" ]]; then
  exit 0
fi

TASK="${TASK}" METHOD="${METHOD}" LR="${SELECTED_LR}" GPU="${GPUS[0]}" OUT_ROOT="${OUT_ROOT}" STAGE=test \
  bash command/run_s2l_human_config.sh \
  2>&1 | tee "${OUT_ROOT}/logs/${METHOD}_selected_lr${SELECTED_LR}.test.log"
