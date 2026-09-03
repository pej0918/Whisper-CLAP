#!/usr/bin/env bash
set -euo pipefail

# Low-level single-configuration launcher for final Human-only S2L experiments.
# Use run_s2l_human_final_protocol.sh for the full 3-LR search.

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
TASK="${TASK:-sent}"
METHOD="${METHOD:-ours}"
GPU="${GPU:-0}"
LR="${LR:-3e-4}"
STAGE="${STAGE:-full}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
SELECTION_BEAM="${SELECTION_BEAM:-5}"
TEST_BEAM="${TEST_BEAM:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

S2L_ROOT="${S2L_ROOT:-/data1/eunju/datasets/speech2latex_asr_human_seed42}"
case "${TASK}" in
  sent) TASK_DIR="${S2L_ROOT}/s2l_sent"; PREFIX="s2l_sent" ;;
  eq)   TASK_DIR="${S2L_ROOT}/s2l_eq";   PREFIX="s2l_eq" ;;
  *) echo "[ERROR] TASK must be sent or eq" >&2; exit 2 ;;
esac

TRAIN_CSV="${TASK_DIR}/${PREFIX}_train.csv"
VAL_CSV="${TASK_DIR}/${PREFIX}_valid.csv"
TEST_CSV="${TASK_DIR}/${PREFIX}_test.csv"
CLAP_EMB="${CLAP_EMB:-${TASK_DIR}/${PREFIX}_clap_text_emb.pt}"
OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/s2l_human_final/${TASK}_seed${SEED}}"

for f in "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}"; do
  [[ -f "${f}" ]] || { echo "[ERROR] Missing ${f}" >&2; exit 1; }
done

case "${STAGE}" in train|test|full) ;; *) echo "[ERROR] STAGE=train|test|full" >&2; exit 2 ;; esac
if [[ "${METHOD}" != "whisper_base" ]]; then
  case "${LR}" in 1e-5|1e-4|3e-4) ;; *) echo "[ERROR] LR must be 1e-5, 1e-4, or 3e-4" >&2; exit 2 ;; esac
fi

if [[ "${METHOD}" == "whisper_base" ]]; then
  OUT_DIR="${OUT_ROOT}/whisper_base"
else
  OUT_DIR="${OUT_ROOT}/${METHOD}_lr${LR}"
fi
mkdir -p "${OUT_DIR}"

run_eval_whisper(){
  local model_path="$1"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper.py \
    --manifest "${TEST_CSV}" --model "${model_path}" \
    --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
    --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --num_beams "${TEST_BEAM}" --max_new_tokens "${MAX_NEW_TOKENS}"
}

run_projector_train(){
  local freeze_whisper="$1" lambda_hidden="$2" lambda_align="$3" align_type="$4"
  local freeze_args=()
  [[ "${freeze_whisper}" == "true" ]] && freeze_args+=(--freeze_whisper)
  [[ -f "${CLAP_EMB}" ]] || { echo "[ERROR] Missing CLAP embedding ${CLAP_EMB}" >&2; exit 1; }

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_mathspeech_projector_best_wer.py \
    --train_csv "${TRAIN_CSV}" --valid_csv "${VAL_CSV}" --test_csv "${TEST_CSV}" \
    --save_dir "${OUT_DIR}" --clap_emb_path "${CLAP_EMB}" --whisper_name "${MODEL}" \
    --adapter_type gated --pool_type cls --adapter_bottleneck 256 \
    --dropout 0.1 --adapter_scale_init 0.01 \
    --align_loss_type "${align_type}" --lambda_align "${lambda_align}" --lambda_hidden "${lambda_hidden}" \
    --batch_size 4 --epochs "${EPOCHS}" --lr "${LR}" --weight_decay 1e-4 \
    --selection_num_beams "${SELECTION_BEAM}" --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" --num_workers "${NUM_WORKERS}" "${freeze_args[@]}"
}

train_method(){
  case "${METHOD}" in
    whisper_base) echo "[INFO] no training" ;;
    fullft)
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_fullft_compat.py \
        --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${OUT_DIR}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --learning_rate "${LR}" --train_batch_size 16 --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
        --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}"
      ;;
    clap_fullft) run_projector_train false 0.1 0.05 cosine ;;
    ours) run_projector_train true 0.1 0.05 cosine ;;
    rq4_ce_only) run_projector_train true 0.0 0.0 none ;;
    rq4_ce_hidden) run_projector_train true 0.1 0.0 none ;;
    rq4_ce_align) run_projector_train true 0.0 0.05 cosine ;;
    lora_noout|lora_out|rq5b_lora_r7)
      rank=32; alpha=32; targets="q_proj,k_proj,v_proj,fc1,fc2"
      [[ "${METHOD}" == "lora_out" ]] && targets="q_proj,k_proj,v_proj,out_proj,fc1,fc2"
      if [[ "${METHOD}" == "rq5b_lora_r7" ]]; then rank=7; alpha=7; fi
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_lora_controlled.py \
        --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "${OUT_DIR}" --model "${MODEL}" \
        --rank "${rank}" --lora_alpha "${alpha}" --target_modules "${targets}" \
        --epochs "${EPOCHS}" --learning_rate "${LR}" --train_batch_size 16 --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
        --generation_num_beams "${SELECTION_BEAM}" --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}"
      ;;
    residual_b256|rq5b_residual_b128)
      bottleneck=256; [[ "${METHOD}" == "rq5b_residual_b128" ]] && bottleneck=128
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/train_whisper_residual_adapter.py \
        --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --save_dir "${OUT_DIR}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --lr "${LR}" --batch_size 4 --num_workers "${NUM_WORKERS}" \
        --adapter_bottleneck "${bottleneck}" --selection_num_beams "${SELECTION_BEAM}" \
        --selection_max_new_tokens "${MAX_NEW_TOKENS}" --seed "${SEED}" --fp16
      ;;
    *) echo "[ERROR] Unknown METHOD=${METHOD}" >&2; exit 2 ;;
  esac
}

test_method(){
  case "${METHOD}" in
    whisper_base) run_eval_whisper "${MODEL}" ;;
    fullft) run_eval_whisper "${OUT_DIR}/best" ;;
    lora_noout|lora_out|rq5b_lora_r7) run_eval_whisper "${OUT_DIR}/merged" ;;
    clap_fullft|ours|rq4_ce_only|rq4_ce_hidden|rq4_ce_align)
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_mathspeech_projector_source_disjoint_hf.py \
        --manifest "${TEST_CSV}" --ckpt "${OUT_DIR}/best.pt" \
        --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
        --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
        --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
        --num_beams "${TEST_BEAM}" --max_new_tokens "${MAX_NEW_TOKENS}" --fp16
      ;;
    residual_b256|rq5b_residual_b128)
      bottleneck=256; [[ "${METHOD}" == "rq5b_residual_b128" ]] && bottleneck=128
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u scripts/eval_whisper_residual_adapter.py \
        --manifest "${TEST_CSV}" --ckpt "${OUT_DIR}/best.pt" --model "${MODEL}" \
        --adapter_bottleneck "${bottleneck}" \
        --output_csv "${OUT_DIR}/test_predictions_beam${TEST_BEAM}.csv" \
        --summary_json "${OUT_DIR}/test_summary_beam${TEST_BEAM}.json" \
        --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
        --num_beams "${TEST_BEAM}" --max_new_tokens "${MAX_NEW_TOKENS}"
      ;;
  esac
}

echo "TASK=${TASK} METHOD=${METHOD} STAGE=${STAGE} LR=${LR} GPU=${GPU}"
if [[ "${STAGE}" == "train" || "${STAGE}" == "full" ]]; then train_method; fi
if [[ "${STAGE}" == "test" || "${STAGE}" == "full" ]]; then test_method; fi
