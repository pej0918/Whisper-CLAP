#!/bin/bash

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments

for EXP in \
v5_ce_residual_mlp_hidden_scale0001 \
v5_ce_gated_hidden_scale0001 \
v5_align_residual_mlp_cosine_lam003_hidden05_scale0001 \
v5_align_residual_mlp_cosine_lam001_hidden05_scale0001 \
v5_ce_residual_mlp_hidden1_scale0001 \
v5_align_residual_mlp_cosine_lam003_hidden1_scale0001
do
  echo "=============================="
  echo "Evaluating $EXP"
  echo "=============================="

  CUDA_VISIBLE_DEVICES=3 python eval_whisper_projector_v2.py \
    --ckpt_path $BASE/$EXP/best.pt \
    --save_csv $BASE/$EXP/result_ASR_test_fixedgen48_clean.csv \
    --eval_split test \
    --max_new_tokens 48
done
