#!/bin/bash
set -euo pipefail

EXP_DIR=${EXP_DIR:-/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter}
SOURCE_COL_ARG=${SOURCE_COL_ARG:-}

python train_whisper_clap_adapter.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --save_dir "$EXP_DIR" \
  --whisper_name openai/whisper-base \
  --seed 42 \
  --force_new_split \
  ${SOURCE_COL_ARG}

python eval_hf_whisper.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --split_path "$EXP_DIR/split_indices.pt" \
  --save_csv "$EXP_DIR/hf_whisper_base_test.csv" \
  --eval_split test \
  --model_name openai/whisper-base

python eval_whisper_clap_adapter.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --ckpt_path "$EXP_DIR/best.pt" \
  --save_csv "$EXP_DIR/clap_adapter_test.csv" \
  --eval_split test

python compute_asr_metrics.py \
  --csv "$EXP_DIR/hf_whisper_base_test.csv" \
  --pred_col pred_hf_whisper_base_test \
  --ref_col transcription \
  --out_csv "$EXP_DIR/hf_whisper_base_test_metrics.csv"

python compute_asr_metrics.py \
  --csv "$EXP_DIR/clap_adapter_test.csv" \
  --ref_col transcription \
  --out_csv "$EXP_DIR/clap_adapter_test_metrics.csv"
