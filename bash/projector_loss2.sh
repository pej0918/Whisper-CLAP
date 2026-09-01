## Loss ablation: 같은 Projector에서 loss만 변경
# Cosine + CLIP
CUDA_VISIBLE_DEVICES=5 python train_whisper_projector_v2.py \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v2_align_residual_mlp_cosine_clip \
  --freeze_whisper \
  --adapter_type residual_mlp \
  --pool_type mean \
  --align_loss_type cosine_clip \
  --lambda_align 0.05 \
  --lambda_cosine 1.0 \
  --lambda_clip 0.1 \
  --temperature 0.07 \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --batch_size 8 \
  --epochs 20

## Pooling ablation: attention pooling

CUDA_VISIBLE_DEVICES=5 python train_whisper_projector_v2.py \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v2_align_gated_attn_cosine \
  --freeze_whisper \
  --adapter_type gated \
  --pool_type attn \
  --align_loss_type cosine \
  --lambda_align 0.1 \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --batch_size 4 \
  --epochs 20