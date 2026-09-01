#!/bin/bash

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
CLAP=/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt

for EXP in \
whisper_base_ft_only \
v5_arch_residual_mlp_ce \
v5_arch_gated_ce \
v5_arch_bottleneck_ce \
v5_arch_linear_residual_ce \
v5_arch_conv1d_ce \
v5_loss_residual_ce \
v5_loss_residual_cosine \
v5_loss_residual_mse \
v5_loss_residual_clip \
v5_loss_residual_cosine_clip \
v5_loss_residual_cosine_mse \
v5_loss_residual_all \
v5_pos_post_encoder_cosine \
v5_pos_encoder_layer3_cosine \
v5_pos_encoder_layer6_cosine \
v5_pos_encoder_layer9_cosine \
v5_pos_both_layer6_cosine \
v5_pool_gated_mean_cosine \
v5_pool_gated_attn_cosine \
v5_pool_gated_cls_cosine
do
  echo "=============================="
  echo "Alignment score: $EXP"
  echo "=============================="

  CUDA_VISIBLE_DEVICES=3 python compute_alignment_score.py \
    --ckpt_path $BASE/$EXP/best.pt \
    --clap_emb_path $CLAP \
    --save_csv $BASE/$EXP/alignment_score_test.csv \
    --eval_split test
done