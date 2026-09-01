#!/bin/bash

# set -e

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
CLAP=/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt

EPOCHS=20
LR=5e-5
BS4=4
BS8=8

COMMON="--freeze_whisper --epochs $EPOCHS --lr $LR --lambda_hidden 0.1 --adapter_scale_init 0.01"

mkdir -p $BASE/logs

echo "=================================================="
echo "Start V5 Ablation Experiments"
echo "=================================================="

run_cmd () {
  local gpu=$1
  local save_dir=$2
  local script=$3
  shift 3

  CUDA_VISIBLE_DEVICES=$gpu python $script \
    --save_dir $save_dir \
    $COMMON \
    "$@" \
    2>&1 | tee ${save_dir}.log
}

# =========================================================
# GPU 3: Architecture Ablation (CE-only)
# =========================================================
(
  echo "[GPU 3] Architecture ablation start"

  run_cmd 3 $BASE/v5_arch_residual_mlp_ce train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --batch_size $BS4

  run_cmd 3 $BASE/v5_arch_gated_ce train_whisper_projector_v2.py \
    --adapter_type gated \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --batch_size $BS4

  run_cmd 3 $BASE/v5_arch_bottleneck_ce train_whisper_projector_v2.py \
    --adapter_type bottleneck \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --batch_size $BS4

  run_cmd 3 $BASE/v5_arch_linear_residual_ce train_whisper_projector_v2.py \
    --adapter_type linear_residual \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --batch_size $BS4

  run_cmd 3 $BASE/v5_arch_conv1d_ce train_whisper_projector_v2.py \
    --adapter_type conv1d \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --batch_size $BS4

  echo "[GPU 3] Architecture ablation done"
) &

# =========================================================
# GPU 4: Loss Ablation
# =========================================================
(
  echo "[GPU 4] Loss ablation start"

  run_cmd 4 $BASE/v5_loss_residual_ce train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type none \
    --lambda_align 0.0 \
    --batch_size $BS4

  run_cmd 4 $BASE/v5_loss_residual_cosine train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 4 $BASE/v5_loss_residual_mse train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type mse \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 4 $BASE/v5_loss_residual_clip train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type clip \
    --lambda_align 0.03 \
    --lambda_clip 1.0 \
    --temperature 0.07 \
    --clap_emb_path $CLAP \
    --batch_size $BS8

  run_cmd 4 $BASE/v5_loss_residual_cosine_clip train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine_clip \
    --lambda_align 0.03 \
    --lambda_cosine 1.0 \
    --lambda_clip 0.1 \
    --temperature 0.07 \
    --clap_emb_path $CLAP \
    --batch_size $BS8

  run_cmd 4 $BASE/v5_loss_residual_cosine_mse train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type cosine_mse \
    --lambda_align 0.05 \
    --lambda_cosine 1.0 \
    --lambda_mse 0.1 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 4 $BASE/v5_loss_residual_all train_whisper_projector_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --align_loss_type all \
    --lambda_align 0.03 \
    --lambda_cosine 1.0 \
    --lambda_mse 0.1 \
    --lambda_clip 0.1 \
    --temperature 0.07 \
    --clap_emb_path $CLAP \
    --batch_size $BS8

  echo "[GPU 4] Loss ablation done"
) &

# =========================================================
# GPU 5: Position Ablation
# =========================================================
(
  echo "[GPU 5] Position ablation start"

  run_cmd 5 $BASE/v5_pos_post_encoder_cosine train_whisper_projector_position_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --adapter_position post_encoder \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 5 $BASE/v5_pos_encoder_layer3_cosine train_whisper_projector_position_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --adapter_position encoder_layer \
    --encoder_layer 3 \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 5 $BASE/v5_pos_encoder_layer6_cosine train_whisper_projector_position_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --adapter_position encoder_layer \
    --encoder_layer 6 \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 5 $BASE/v5_pos_encoder_layer9_cosine train_whisper_projector_position_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --adapter_position encoder_layer \
    --encoder_layer 9 \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 5 $BASE/v5_pos_both_layer6_cosine train_whisper_projector_position_v2.py \
    --adapter_type residual_mlp \
    --pool_type mean \
    --adapter_position both \
    --encoder_layer 6 \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  echo "[GPU 5] Position ablation done"
) &

# =========================================================
# GPU 2: Pooling Ablation
# =========================================================
(
  echo "[GPU 2] Pooling ablation start"

  run_cmd 2 $BASE/v5_pool_gated_mean_cosine train_whisper_projector_v2.py \
    --adapter_type gated \
    --pool_type mean \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 2 $BASE/v5_pool_gated_attn_cosine train_whisper_projector_v2.py \
    --adapter_type gated \
    --pool_type attn \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  run_cmd 2 $BASE/v5_pool_gated_cls_cosine train_whisper_projector_v2.py \
    --adapter_type gated \
    --pool_type cls \
    --align_loss_type cosine \
    --lambda_align 0.05 \
    --clap_emb_path $CLAP \
    --batch_size $BS4

  echo "[GPU 2] Pooling ablation done"
) &

wait

echo "=================================================="
echo "All projector ablation experiments finished."
echo "Logs saved to: $BASE/logs"
echo "=================================================="