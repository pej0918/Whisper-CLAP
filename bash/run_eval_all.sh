#!/bin/bash

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments

for EXP in \
v2_ce_residual_mlp \
v2_ce_gated \
v2_ce_bottleneck \
v2_align_residual_mlp_cosine \
v2_align_residual_mlp_clip \
v2_align_residual_mlp_cosine_clip \
v2_align_gated_attn_cosine
do
  echo "=============================="
  echo "Evaluating $EXP"
  echo "=============================="

  python eval_whisper_projector_v2.py \
    --ckpt_path $BASE/$EXP/best.pt \
    --save_csv $BASE/$EXP/result_ASR_test_fixedgen.csv \
    --eval_split test \
    --max_new_tokens 48
done