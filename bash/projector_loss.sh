## Loss ablation: 같은 Projector에서 loss만 변경
# Cosine alignment
CUDA_VISIBLE_DEVICES=4 python train_whisper_projector_v2.py \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v2_align_residual_mlp_cosine \
  --freeze_whisper \
  --adapter_type residual_mlp \
  --pool_type mean \
  --align_loss_type cosine \
  --lambda_align 0.1 \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --batch_size 4 \
  --epochs 20

# In-batch contrastive / CLIP-style
CUDA_VISIBLE_DEVICES=4 python train_whisper_projector_v2.py \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v2_align_residual_mlp_clip \
  --freeze_whisper \
  --adapter_type residual_mlp \
  --pool_type mean \
  --align_loss_type clip \
  --lambda_align 0.05 \
  --temperature 0.07 \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --batch_size 8 \
  --epochs 20