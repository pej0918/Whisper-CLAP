## Projector 구조 ablation : CE-only
CUDA_VISIBLE_DEVICES=3 python train_whisper_projector_v2.py \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v2_ce_residual_mlp \
  --freeze_whisper \
  --adapter_type residual_mlp \
  --pool_type mean \
  --align_loss_type none \
  --lambda_align 0.0 \
  --batch_size 4 \
  --epochs 20


CUDA_VISIBLE_DEVICES=3 python train_whisper_projector_v2.py \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v2_ce_gated \
  --freeze_whisper \
  --adapter_type gated \
  --pool_type mean \
  --align_loss_type none \
  --lambda_align 0.0 \
  --batch_size 4 \
  --epochs 20

CUDA_VISIBLE_DEVICES=3 python train_whisper_projector_v2.py \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/v2_ce_bottleneck \
  --freeze_whisper \
  --adapter_type bottleneck \
  --pool_type mean \
  --align_loss_type none \
  --lambda_align 0.0 \
  --batch_size 4 \
  --epochs 20