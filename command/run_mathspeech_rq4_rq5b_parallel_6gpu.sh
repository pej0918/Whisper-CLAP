#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# MathSpeech RQ4 + RQ5-B controlled experiments
# ============================================================================
# Common protocol:
#   Backbone          : openai/whisper-base
#   Epochs            : 10
#   LR search         : {1e-5, 1e-4}
#   Checkpoint select : lowest validation WER with beam=5
#   Main test decode  : beam=5
#   Seed              : 42
#
# RQ4 loss ablation (same Ours architecture, frozen Whisper):
#   1) CE only
#   2) CE + Hidden
#   3) CE + Align
#   Full Ours (CE + Hidden + Align) is reused from the existing common grid.
#
# RQ5-B parameter-matched PEFT:
#   4) LoRA -out_proj, r=7, alpha=7  (~0.817M)
#   5) Residual Adapter, bottleneck=128 (790,272 params)
#   Ours = 790,273 params.
#
# Run:
#   GPUS="0 1 2 3 4 5" bash command/run_mathspeech_rq4_rq5b_parallel_6gpu.sh
# ============================================================================

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
LRS_STR="${LRS:-1e-5 1e-4}"
SELECTION_BEAM="${SELECTION_BEAM:-5}"
TEST_BEAM="${TEST_BEAM:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"

OURS_BATCH_SIZE="${OURS_BATCH_SIZE:-4}"
LORA_BATCH_SIZE="${LORA_BATCH_SIZE:-16}"
RESIDUAL_BATCH_SIZE="${RESIDUAL_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

GPUS_STR="${GPUS:-0 1 2 3 4 5}"
read -r -a GPU_LIST <<< "${GPUS_STR}"
if [[ ${#GPU_LIST[@]} -lt 6 ]]; then
  echo "[ERROR] Need 6 GPU ids in GPUS, e.g. GPUS=\"0 1 2 3 4 5\"" >&2
  exit 2
fi

read -r -a LRS <<< "${LRS_STR}"
if [[ ${#LRS[@]} -ne 2 ]]; then
  echo "[ERROR] Expected exactly two LRs; got: ${LRS[*]}" >&2
  exit 2
fi
LR1="${LRS[0]}"
LR2="${LRS[1]}"

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
OURS_TRAIN_CSV="${OURS_TRAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
OURS_VAL_CSV="${OURS_VAL_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
OURS_TEST_CSV="${OURS_TEST_CSV:-${SPLIT_DIR}/mathspeech_test_source_seed42.csv}"
CLAP_EMB="${CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"

OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_rq4_rq5b_bestvalidwer_seed${SEED}_6gpu}"
mkdir -p "${OUT_ROOT}/logs"

require_file(){
  [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }
}
for f in \
  "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}" \
  "${OURS_TRAIN_CSV}" "${OURS_VAL_CSV}" "${OURS_TEST_CSV}" \
  "${CLAP_EMB}"; do
  require_file "$f"
done

cat > "${OUT_ROOT}/protocol.txt" <<EOF
MODEL=${MODEL}
SEED=${SEED}
EPOCHS=${EPOCHS}
LRS=${LRS[*]}
SELECTION_BEAM=${SELECTION_BEAM}
TEST_BEAM=${TEST_BEAM}
RQ4_ARCHITECTURE=gated,bottleneck256,pool=first_encoder_timestep,scale_init=0.01,Whisper_frozen
RQ4_LAMBDA_ALIGN=0.05 when enabled
RQ4_LAMBDA_HIDDEN=0.1 when enabled
RQ5B_LORA=rank7,alpha7,q_proj,k_proj,v_proj,fc1,fc2
RQ5B_RESIDUAL=bottleneck128
EOF

echo "================================================================================"
echo "MATHSPEECH RQ4 + RQ5-B"
echo "================================================================================"
echo "OUT_ROOT       : ${OUT_ROOT}"
echo "GPUS           : ${GPU_LIST[*]}"
echo "EPOCHS         : ${EPOCHS}"
echo "LRS            : ${LRS[*]}"
echo "SELECTION BEAM : ${SELECTION_BEAM}"
echo "TEST BEAM      : ${TEST_BEAM}"
echo "================================================================================"

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
run_eval_projector(){
  local gpu="$1" ckpt="$2" out_dir="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_mathspeech_projector_source_disjoint_hf.py \
    --manifest "${OURS_TEST_CSV}" \
    --ckpt "${ckpt}" \
    --output_csv "${out_dir}/test_predictions_beam${TEST_BEAM}.csv" \
    --summary_json "${out_dir}/test_summary_beam${TEST_BEAM}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${TEST_BEAM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --fp16 \
    2>&1 | tee "${out_dir}/eval_beam${TEST_BEAM}.log"
}

run_eval_whisper(){
  local gpu="$1" model_path="$2" out_dir="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_whisper.py \
    --manifest "${TEST_CSV}" \
    --model "${model_path}" \
    --output_csv "${out_dir}/test_predictions_beam${TEST_BEAM}.csv" \
    --summary_json "${out_dir}/test_summary_beam${TEST_BEAM}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${TEST_BEAM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${TEST_BEAM}.log"
}

run_eval_residual(){
  local gpu="$1" ckpt="$2" out_dir="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
    --manifest "${TEST_CSV}" \
    --ckpt "${ckpt}" \
    --model "${MODEL}" \
    --adapter_bottleneck 128 \
    --output_csv "${out_dir}/test_predictions_beam${TEST_BEAM}.csv" \
    --summary_json "${out_dir}/test_summary_beam${TEST_BEAM}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${TEST_BEAM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${TEST_BEAM}.log"
}

# ---------------------------------------------------------------------------
# RQ4: identical architecture; only lambda_hidden / lambda_align change.
# ---------------------------------------------------------------------------
run_rq4_job(){
  local gpu="$1" lr="$2" variant="$3" lambda_hidden="$4" lambda_align="$5" align_type="$6"
  local out_dir="${OUT_ROOT}/rq4_${variant}_lr${lr}"
  mkdir -p "${out_dir}"

  echo "[RQ4] variant=${variant} gpu=${gpu} lr=${lr} hidden=${lambda_hidden} align=${lambda_align} type=${align_type}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_mathspeech_projector_best_wer.py \
    --train_csv "${OURS_TRAIN_CSV}" \
    --valid_csv "${OURS_VAL_CSV}" \
    --test_csv "${OURS_TEST_CSV}" \
    --save_dir "${out_dir}" \
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
    --batch_size "${OURS_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${lr}" \
    --weight_decay 1e-4 \
    --selection_num_beams "${SELECTION_BEAM}" \
    --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --freeze_whisper \
    2>&1 | tee "${out_dir}/train.log"

  run_eval_projector "${gpu}" "${out_dir}/best.pt" "${out_dir}"
}

# ---------------------------------------------------------------------------
# RQ5-B: parameter-matched PEFT baselines.
# ---------------------------------------------------------------------------
run_lora_r7_job(){
  local gpu="$1" lr="$2"
  local out_dir="${OUT_ROOT}/rq5b_lora_r7_nooutproj_lr${lr}"
  mkdir -p "${out_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_lora_controlled.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --output_dir "${out_dir}" \
    --model "${MODEL}" \
    --rank 7 \
    --lora_alpha 7 \
    --target_modules "q_proj,k_proj,v_proj,fc1,fc2" \
    --epochs "${EPOCHS}" \
    --learning_rate "${lr}" \
    --train_batch_size "${LORA_BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --num_workers "${NUM_WORKERS}" \
    --generation_num_beams "${SELECTION_BEAM}" \
    --generation_max_length "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    2>&1 | tee "${out_dir}/train.log"

  run_eval_whisper "${gpu}" "${out_dir}/merged" "${out_dir}"
}

run_residual_b128_job(){
  local gpu="$1" lr="$2"
  local out_dir="${OUT_ROOT}/rq5b_residual_b128_lr${lr}"
  mkdir -p "${out_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
    --train "${TRAIN_CSV}" \
    --dev "${VAL_CSV}" \
    --save_dir "${out_dir}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --lr "${lr}" \
    --batch_size "${RESIDUAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --adapter_bottleneck 128 \
    --selection_num_beams "${SELECTION_BEAM}" \
    --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --fp16 \
    2>&1 | tee "${out_dir}/train.log"

  run_eval_residual "${gpu}" "${out_dir}/best.pt" "${out_dir}"
}

# ---------------------------------------------------------------------------
# Parallel launcher
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
  pids=()
  names=()
  [[ ${status} -eq 0 ]] || { echo "[ERROR] Wave failed" >&2; exit ${status}; }
}

# Wave 1: RQ4 = 3 variants x 2 learning rates = 6 jobs.
launch "rq4_ce_only_lr${LR1}_gpu${GPU_LIST[0]}" \
  run_rq4_job "${GPU_LIST[0]}" "${LR1}" ce_only 0.0 0.0 none
launch "rq4_ce_only_lr${LR2}_gpu${GPU_LIST[1]}" \
  run_rq4_job "${GPU_LIST[1]}" "${LR2}" ce_only 0.0 0.0 none

launch "rq4_ce_hidden_lr${LR1}_gpu${GPU_LIST[2]}" \
  run_rq4_job "${GPU_LIST[2]}" "${LR1}" ce_hidden 0.1 0.0 none
launch "rq4_ce_hidden_lr${LR2}_gpu${GPU_LIST[3]}" \
  run_rq4_job "${GPU_LIST[3]}" "${LR2}" ce_hidden 0.1 0.0 none

launch "rq4_ce_align_lr${LR1}_gpu${GPU_LIST[4]}" \
  run_rq4_job "${GPU_LIST[4]}" "${LR1}" ce_align 0.0 0.05 cosine
launch "rq4_ce_align_lr${LR2}_gpu${GPU_LIST[5]}" \
  run_rq4_job "${GPU_LIST[5]}" "${LR2}" ce_align 0.0 0.05 cosine

wait_wave

# Wave 2: RQ5-B = 2 baselines x 2 learning rates = 4 jobs.
launch "rq5b_lora_r7_lr${LR1}_gpu${GPU_LIST[0]}" \
  run_lora_r7_job "${GPU_LIST[0]}" "${LR1}"
launch "rq5b_lora_r7_lr${LR2}_gpu${GPU_LIST[1]}" \
  run_lora_r7_job "${GPU_LIST[1]}" "${LR2}"

launch "rq5b_residual_b128_lr${LR1}_gpu${GPU_LIST[2]}" \
  run_residual_b128_job "${GPU_LIST[2]}" "${LR1}"
launch "rq5b_residual_b128_lr${LR2}_gpu${GPU_LIST[3]}" \
  run_residual_b128_job "${GPU_LIST[3]}" "${LR2}"

wait_wave

"${PYTHON}" -u scripts/collect_mathspeech_rq4_rq5b_results.py \
  --output_dir "${OUT_ROOT}"

echo "================================================================================"
echo "DONE"
echo "Results: ${OUT_ROOT}"
echo "================================================================================"
