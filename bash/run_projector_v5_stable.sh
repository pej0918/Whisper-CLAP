#!/bin/bash

set -e

PROJECT_DIR=/home/pej0918/Projects/Audio_Text
EXP_DIR=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
CLAP_EMB=/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt
TRAIN_SCRIPT=train_whisper_projector_v2.py

cd $PROJECT_DIR
mkdir -p $EXP_DIR/logs

echo "=========================================="
echo "Start v5 stable projector experiments"
echo "Goal: reduce decoder disruption / tail hallucination"
echo "=========================================="


# =========================================================
# GPU 3: CE-only stability baselines
# =========================================================
(
  echo "[GPU 3] CE-only stable baselines"

  CUDA_VISIBLE_DEVICES=3 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v5_ce_residual_mlp_hidden_scale0001 \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --lambda_hidden 0.5 \
    --adapter_scale_init 0.001 \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v5_ce_residual_mlp_hidden_scale0001.log

  CUDA_VISIBLE_DEVICES=3 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v5_ce_gated_hidden_scale0001 \
    --freeze_whisper \
    --adapter_type gated \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --lambda_hidden 0.5 \
    --adapter_scale_init 0.001 \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v5_ce_gated_hidden_scale0001.log

  echo "[GPU 3] done"
) &


# =========================================================
# GPU 4: CLAP cosine alignment with weaker intervention
# =========================================================
(
  echo "[GPU 4] cosine alignment stable variants"

  CUDA_VISIBLE_DEVICES=4 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v5_align_residual_mlp_cosine_lam003_hidden05_scale0001 \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.03 \
    --lambda_hidden 0.5 \
    --adapter_scale_init 0.001 \
    --clap_emb_path $CLAP_EMB \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v5_align_residual_mlp_cosine_lam003_hidden05_scale0001.log

  CUDA_VISIBLE_DEVICES=4 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v5_align_residual_mlp_cosine_lam001_hidden05_scale0001 \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.01 \
    --lambda_hidden 0.5 \
    --adapter_scale_init 0.001 \
    --clap_emb_path $CLAP_EMB \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v5_align_residual_mlp_cosine_lam001_hidden05_scale0001.log

  echo "[GPU 4] done"
) &


# =========================================================
# GPU 5: very conservative intervention
# =========================================================
(
  echo "[GPU 5] ultra-stable variants"

  CUDA_VISIBLE_DEVICES=5 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v5_ce_residual_mlp_hidden1_scale0001 \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --lambda_hidden 1.0 \
    --adapter_scale_init 0.001 \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v5_ce_residual_mlp_hidden1_scale0001.log

  CUDA_VISIBLE_DEVICES=5 python $TRAIN_SCRIPT \
    --save_dir $EXP_DIR/v5_align_residual_mlp_cosine_lam003_hidden1_scale0001 \
    --freeze_whisper \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.03 \
    --lambda_hidden 1.0 \
    --adapter_scale_init 0.001 \
    --clap_emb_path $CLAP_EMB \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    2>&1 | tee $EXP_DIR/logs/v5_align_residual_mlp_cosine_lam003_hidden1_scale0001.log

  echo "[GPU 5] done"
) &


wait

echo "=========================================="
echo "All v5 stable experiments finished."
echo "Logs saved to: $EXP_DIR/logs"
echo "=========================================="
