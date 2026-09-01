#!/bin/bash

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments

for EXP in \
v4_ce_residual_mlp_hidden \
v4_ce_gated_hidden \
v4_ce_bottleneck_hidden \
v4_align_residual_mlp_cosine_hidden \
v4_align_residual_mlp_clip_hidden \
v4_align_residual_mlp_cosine_clip_hidden \
v4_align_gated_attn_cosine_hidden
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
