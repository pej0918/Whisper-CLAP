# #!/usr/bin/env bash
# set -euo pipefail

# # =============================================
# # Gated Projector Ablations (GPU 3,4,5)
# # - Alignment Loss Ablation
# # - Projector Position Ablation
# # =============================================

# BASE="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments"
# CLAP="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt"
# # TRAIN_SCRIPT="train_whisper_projector_v2.py"
# # ABALATION_SCRIPT="/home/pej0918/Projects/Audio_Text/train_whisper_projector_ablation.py"
# # 공통 학습 설정
# EPOCHS=20
# BATCH_SIZE=4
# LR="5e-5"
# LAMBDA_HIDDEN="0.1"
# LAMBDA_ALIGN_COS_MSE="0.05"
# LAMBDA_ALIGN_CLIP="0.03"

# # GPU pool
# GPUS=(3 4 5)
# MAX_JOBS=3

# mkdir -p "$BASE"

# run_train() {
#   local gpu=$1
#   local save_dir=$2
#   shift 2

#   echo "============================================================"
#   echo "[START] GPU=${gpu} SAVE_DIR=${save_dir}"
#   echo "============================================================"

#   CUDA_VISIBLE_DEVICES=${gpu} python train_whisper_projector_ablation.py \
#     --save_dir "$save_dir" \
#     --adapter_type gated \
#     --pool_type mean \
#     --lambda_hidden "$LAMBDA_HIDDEN" \
#     --freeze_whisper \
#     --epochs "$EPOCHS" \
#     --batch_size "$BATCH_SIZE" \
#     --lr "$LR" \
#     --fp16 \
#     "$@"

#   echo "============================================================"
#   echo "[DONE] GPU=${gpu} SAVE_DIR=${save_dir}"
#   echo "============================================================"
# }

# wait_batch() {
#   echo "Waiting for current batch..."
#   wait
#   echo "Batch finished."
# }

# # ============================================================
# # 1) Alignment Loss Ablation with Gated Projector
# #    고정: adapter=gated, pool=mean, position=post_encoder
# # ============================================================

# run_train 3 "$BASE/gated_loss_ce" \
#   --adapter_position post_encoder \
#   --align_loss_type none \
#   --lambda_align 0.0 &

# run_train 4 "$BASE/gated_loss_cosine" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position post_encoder \
#   --align_loss_type cosine \
#   --lambda_align "$LAMBDA_ALIGN_COS_MSE" &

# run_train 5 "$BASE/gated_loss_mse" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position post_encoder \
#   --align_loss_type mse \
#   --lambda_align "$LAMBDA_ALIGN_COS_MSE" &

# wait_batch

# run_train 3 "$BASE/gated_loss_clip" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position post_encoder \
#   --align_loss_type clip \
#   --lambda_align "$LAMBDA_ALIGN_CLIP" &

# run_train 4 "$BASE/gated_loss_cosine_clip" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position post_encoder \
#   --align_loss_type cosine_clip \
#   --lambda_align "$LAMBDA_ALIGN_CLIP" \
#   --lambda_cosine 1.0 \
#   --lambda_clip 0.1 &

# wait_batch

# # ============================================================
# # 2) Projector Position Ablation with Gated Projector
# #    고정: adapter=gated, pool=mean, loss=cosine
# # ============================================================

# run_train 3 "$BASE/gated_pos_post_encoder_cosine" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position post_encoder \
#   --align_loss_type cosine \
#   --lambda_align "$LAMBDA_ALIGN_COS_MSE" &

# run_train 4 "$BASE/gated_pos_encoder_layer3_cosine" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position encoder_layer \
#   --encoder_layer 3 \
#   --align_loss_type cosine \
#   --lambda_align "$LAMBDA_ALIGN_COS_MSE" &

# run_train 5 "$BASE/gated_pos_encoder_layer6_cosine" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position encoder_layer \
#   --encoder_layer 6 \
#   --align_loss_type cosine \
#   --lambda_align "$LAMBDA_ALIGN_COS_MSE" &

# wait_batch

# run_train 3 "$BASE/gated_pos_encoder_layer9_cosine" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position encoder_layer \
#   --encoder_layer 9 \
#   --align_loss_type cosine \
#   --lambda_align "$LAMBDA_ALIGN_COS_MSE" &

# run_train 4 "$BASE/gated_pos_both_layer6_cosine" \
#   --clap_emb_path "$CLAP" \
#   --adapter_position both \
#   --encoder_layer 6 \
#   --align_loss_type cosine \
#   --lambda_align "$LAMBDA_ALIGN_COS_MSE" &

# wait_batch

# echo "All gated ablation experiments finished."


# !/usr/bin/env bash
# set -euo pipefail

# # =============================
# # Eval script for Gated ablations
# # GPU: 3, 4, 5
# # =============================

# BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
# EVAL_SCRIPT=eval_whisper_projector_any_v2.py

# # output csv name
# OUT_CSV=result_ASR_test_fixedgen48_clean.csv

# # generation setting
# MAX_NEW_TOKENS=48
# EVAL_SPLIT=test

# # use these GPUs in round-robin
# GPUS=(3 4 5)
# MAX_JOBS=${#GPUS[@]}

# # Experiments to evaluate
# EXPS=(
#   # Alignment loss ablation with Gated projector
#   gated_loss_ce
#   gated_loss_cosine
#   gated_loss_mse
#   gated_loss_clip
#   gated_loss_cosine_clip

#   # Projector position ablation with Gated projector
#   gated_pos_post_encoder_cosine
#   gated_pos_encoder_layer3_cosine
#   gated_pos_encoder_layer6_cosine
#   gated_pos_encoder_layer9_cosine
#   gated_pos_both_layer6_cosine
# )

# run_eval () {
#   local gpu=$1
#   local exp=$2
#   local ckpt=${BASE}/${exp}/best.pt
#   local save_csv=${BASE}/${exp}/${OUT_CSV}

#   if [ ! -f "$ckpt" ]; then
#     echo "[SKIP] checkpoint not found: $ckpt"
#     return 0
#   fi

#   echo "========================================"
#   echo "[EVAL] GPU=$gpu  EXP=$exp"
#   echo "ckpt: $ckpt"
#   echo "save: $save_csv"
#   echo "========================================"

#   CUDA_VISIBLE_DEVICES=$gpu python $EVAL_SCRIPT \
#     --ckpt_path "$ckpt" \
#     --save_csv "$save_csv" \
#     --eval_split "$EVAL_SPLIT" \
#     --max_new_tokens "$MAX_NEW_TOKENS"
# }

# job_count=0
# for i in "${!EXPS[@]}"; do
#   gpu=${GPUS[$((i % MAX_JOBS))]}
#   exp=${EXPS[$i]}

#   run_eval "$gpu" "$exp" &
#   job_count=$((job_count + 1))

#   # wait every 3 jobs
#   if (( job_count % MAX_JOBS == 0 )); then
#     wait
#   fi
# done

# wait

# echo "All gated ablation evaluations finished."

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments

# METRIC_SCRIPT=compute_asr_metrics.py



# EXPS=(

#   gated_loss_ce

#   gated_loss_cosine

#   gated_loss_mse

#   gated_loss_clip

#   gated_loss_cosine_clip

#   gated_pos_post_encoder_cosine

#   gated_pos_encoder_layer3_cosine

#   gated_pos_encoder_layer6_cosine

#   gated_pos_encoder_layer9_cosine

#   gated_pos_both_layer6_cosine

# )



# for EXP in "${EXPS[@]}"; do

#   echo "=============================="

#   echo "Computing metrics: $EXP"

#   echo "=============================="



#   python compute_asr_metrics_v4.py \
#     --csv $BASE/$EXP/result_ASR_test_fixedgen48_clean.csv \
#     --out_csv $BASE/$EXP/metrics_test_clean.csv

# done

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
CLAP=/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt

for EXP in \
gated_loss_ce \
gated_loss_cosine \
gated_loss_mse \
gated_loss_clip \
gated_loss_cosine_clip \
gated_pos_post_encoder_cosine \
gated_pos_encoder_layer3_cosine \
gated_pos_encoder_layer6_cosine \
gated_pos_encoder_layer9_cosine \
gated_pos_both_layer6_cosine
do
  echo "==================== $EXP ===================="
  CUDA_VISIBLE_DEVICES=3 python compute_align_345.py \
    --ckpt_path $BASE/$EXP/best.pt \
    --clap_emb_path $CLAP \
    --eval_split test \
    --batch_size 16
done