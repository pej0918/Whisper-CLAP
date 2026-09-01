#!/bin/bash

set -e

PROJECT_DIR=/home/pej0918/Projects/Audio_Text
EXP_DIR=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
CLAP_EMB=/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt
TRAIN_SCRIPT=train_whisper_projector_v2.py

cd $PROJECT_DIR

mkdir -p $EXP_DIR/logs

echo "=========================================="
echo "Start Projector Ablation Experiments"
echo "Project dir: $PROJECT_DIR"
echo "Experiment dir: $EXP_DIR"
echo "=========================================="


# =========================================================
# GPU 3: Projector structure ablation, CE-only
# =========================================================
(
  echo "=========================================="
  echo "[GPU 3] CE-only projector ablation start"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES=3 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v4_ce_residual_mlp_hidden \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --lambda_hidden 0.1 \
    --adapter_scale_init 0.01 \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v4_ce_residual_mlp_hidden.log

  CUDA_VISIBLE_DEVICES=3 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v4_ce_gated_hidden \
    --freeze_whisper \
    --adapter_type gated \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --lambda_hidden 0.1 \
    --adapter_scale_init 0.01 \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v4_ce_gated_hidden.log

  CUDA_VISIBLE_DEVICES=3 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v4_ce_bottleneck_hidden \
    --freeze_whisper \
    --adapter_type bottleneck \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --lambda_hidden 0.1 \
    --adapter_scale_init 0.01 \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v4_ce_bottleneck_hidden.log

  echo "[GPU 3] CE-only projector ablation done"
) &


# =========================================================
# GPU 4: Loss ablation, residual MLP
# =========================================================
(
  echo "=========================================="
  echo "[GPU 4] Loss ablation start"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES=4 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v4_align_residual_mlp_cosine_hidden \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --lambda_hidden 0.1 \
    --adapter_scale_init 0.01 \
    --clap_emb_path $CLAP_EMB \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v4_align_residual_mlp_cosine_hidden.log

  CUDA_VISIBLE_DEVICES=4 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v4_align_residual_mlp_clip_hidden \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type clip \
    --lambda_align 0.03 \
    --lambda_hidden 0.1 \
    --adapter_scale_init 0.01 \
    --temperature 0.07 \
    --clap_emb_path $CLAP_EMB \
    --batch_size 8 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v4_align_residual_mlp_clip_hidden.log

  echo "[GPU 4] Loss ablation done"
) &


# =========================================================
# GPU 5: Cosine+CLIP and pooling ablation
# =========================================================
(
  echo "=========================================="
  echo "[GPU 5] Cosine+CLIP / pooling ablation start"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES=5 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v4_align_residual_mlp_cosine_clip_hidden \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine_clip \
    --lambda_align 0.03 \
    --lambda_hidden 0.1 \
    --lambda_cosine 1.0 \
    --lambda_clip 0.1 \
    --adapter_scale_init 0.01 \
    --temperature 0.07 \
    --clap_emb_path $CLAP_EMB \
    --batch_size 8 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v4_align_residual_mlp_cosine_clip_hidden.log

  CUDA_VISIBLE_DEVICES=5 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v4_align_gated_attn_cosine_hidden \
    --freeze_whisper \
    --adapter_type gated \
    --pool_type attn \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --lambda_hidden 0.1 \
    --adapter_scale_init 0.01 \
    --clap_emb_path $CLAP_EMB \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v4_align_gated_attn_cosine_hidden.log

  echo "[GPU 5] Cosine+CLIP / pooling ablation done"
) &


wait

echo "=========================================="
echo "All projector ablation experiments finished."
echo "Logs saved to: $EXP_DIR/logs"
echo "=========================================="
