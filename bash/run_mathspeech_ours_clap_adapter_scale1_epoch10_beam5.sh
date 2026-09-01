#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT="/home/pej0918/Projects/Audio_Text"
DATA_XLSX="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
AUDIO_DIR="/data1/eunju/datasets/mathspeech/dataset"
CLAP_EMB="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt"
EXP_ROOT="${ROOT}/MathSpeech/Experiments"
SPLIT_PATH="${EXP_ROOT}/source_aware_seed42/split_indices.pt"
SAVE_DIR="${EXP_ROOT}/epoch10_beam5_clap_adapter_frozen_align010"
LOG_DIR="${EXP_ROOT}/logs_all_methods_epoch10_beam5"
LOG_FILE="${LOG_DIR}/04_ours_clap_adapter.log"

EPOCHS=10
LR="1e-5"
BATCH_SIZE=4
NUM_WORKERS=2
SEED=42
NUM_BEAMS=5
MAX_NEW_TOKENS=256
SELECTION_MAX_NEW_TOKENS=256

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"
cd "${ROOT}"

{
  echo "[Ours-Scale1] CLAP-guided adapter, Whisper frozen"
  echo "  backbone           : openai/whisper-base"
  echo "  split              : ${SPLIT_PATH}"
  echo "  lr                 : ${LR}"
  echo "  epochs             : ${EPOCHS}"
  echo "  adapter_type       : gated"
  echo "  adapter_bottleneck : 256"
  echo "  adapter_scale_init : 1.0"
  echo "  lambda_align       : 0.10"
  echo "  lambda_hidden      : 0.10"
  echo "  eval beams         : ${NUM_BEAMS}"

  python train_whisper_clap_adapter.py \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --clap_emb_path "${CLAP_EMB}" \
    --split_path "${SPLIT_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --whisper_name openai/whisper-base \
    --text_col transcription \
    --source_col Source \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --adapter_type gated \
    --adapter_bottleneck 256 \
    --adapter_scale_init 1.0 \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.10 \
    --lambda_hidden 0.10 \
    --freeze_whisper \
    --num_workers "${NUM_WORKERS}" \
    --selection_max_new_tokens "${SELECTION_MAX_NEW_TOKENS}"

  python eval_whisper.py \
    --model_kind clap_adapter \
    --dataset_type mathspeech \
    --excel_path "${DATA_XLSX}" \
    --audio_dir "${AUDIO_DIR}" \
    --split_path "${SPLIT_PATH}" \
    --ckpt_path "${SAVE_DIR}/best.pt" \
    --output_csv "${SAVE_DIR}/test.csv" \
    --summary_json "${SAVE_DIR}/summary.json" \
    --eval_split test \
    --pred_col pred_ours_clap_adapter_test \
    --whisper_name openai/whisper-base \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"

  python compute_asr_metrics.py \
    --csv "${SAVE_DIR}/test.csv" \
    --pred_col pred_ours_clap_adapter_test \
    --ref_col transcription \
    --out_csv "${SAVE_DIR}/test_metrics.csv"

  echo "[Ours-Scale1] Done"
} 2>&1 | tee "${LOG_FILE}"
