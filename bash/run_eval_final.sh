# #!/bin/bash

# BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments

# for EXP in \
# whisper_base_ft_only \
# v5_arch_residual_mlp_ce \
# v5_arch_gated_ce \
# v5_arch_bottleneck_ce \
# v5_arch_linear_residual_ce \
# v5_arch_conv1d_ce \
# v5_loss_residual_ce \
# v5_loss_residual_cosine \
# v5_loss_residual_mse \
# v5_loss_residual_clip \
# v5_loss_residual_cosine_clip \
# v5_loss_residual_cosine_mse \
# v5_loss_residual_all \
# v5_pos_post_encoder_cosine \
# v5_pos_encoder_layer3_cosine \
# v5_pos_encoder_layer6_cosine \
# v5_pos_encoder_layer9_cosine \
# v5_pos_both_layer6_cosine \
# v5_pool_gated_mean_cosine \
# v5_pool_gated_attn_cosine \
# v5_pool_gated_cls_cosine
# do
#   echo "=============================="
#   echo "Evaluating $EXP"
#   echo "=============================="

#   CUDA_VISIBLE_DEVICES=3 python eval_whisper_projector_any_v2.py \
#     --ckpt_path $BASE/$EXP/best.pt \
#     --save_csv $BASE/$EXP/result_ASR_test_fixedgen48_clean.csv \
#     --eval_split test \
#     --max_new_tokens 48
# done


#!/bin/bash

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments

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
  echo "Metrics: $EXP"
  echo "=============================="

  CSV=$BASE/$EXP/result_ASR_test_fixedgen48_clean.csv

  if [ ! -f "$CSV" ]; then
    echo "[missing] $CSV"
    continue
  fi

  python compute_asr_metrics.py \
    --input_csv $CSV \
    --gt_col transcription \
    --save_prefix $BASE/$EXP/metrics_test
done