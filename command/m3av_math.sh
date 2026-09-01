# whisper-base
CUDA_VISIBLE_DEVICES=<빈GPU> python -u eval_whisper.py \
  --model openai/whisper-base \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_math_test.csv \
  --output_csv results/base/math_test_predictions_beam5.csv \
  --summary_json results/base/math_test_summary_beam5.json \
  --batch_size 16 \
  --num_beams 5 \
  --max_new_tokens 256

# full ft train
CUDA_VISIBLE_DEVICES=5 nohup python -u train_whisper_fullft.py \
  --train /data1/dohee/datasets/M3AV/manifests/m3av_math_train.csv \
  --dev /data1/dohee/datasets/M3AV/manifests/m3av_math_dev.csv \
  --output_dir results/full_ft/math \
  --epochs 3 \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  > results/full_ft/math_train.log 2>&1 &


# full ft test
CUDA_VISIBLE_DEVICES=4 python eval_whisper.py \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_math_test.csv \
  --output_csv results/full_ft/math_test.csv \
  --summary_json results/full_ft/math_test.json \
  --model results/full_ft/math/best \
  --batch_size 16 \
  --num_beams 5


# LoRA train
CUDA_VISIBLE_DEVICES=5 nohup python -u train_whisper_lora.py \
  --train /data1/dohee/datasets/M3AV/manifests/m3av_math_train.csv \
  --dev /data1/dohee/datasets/M3AV/manifests/m3av_math_dev.csv \
  --output_dir results/lora/math \
  --epochs 10 \
  --learning_rate 1e-4 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  > results/lora/math_train.log 2>&1 &

# LoRA test
CUDA_VISIBLE_DEVICES=4 python eval_whisper.py \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_math_test.csv \
  --output_csv results/lora/math_test.csv \
  --summary_json results/lora/math_test.json \
  --model results/lora/math/merged \
  --batch_size 16 \
  --num_beams 5


# ours train
CUDA_VISIBLE_DEVICES=5 nohup bash -c '
for cfg in "005 0.05" "010 0.10"; do
    set -- $cfg
    tag=$1
    lam=$2

    echo "========================================"
    echo "Training Math Ours lambda=${lam}"
    echo "========================================"

    python -u train_whisper_ours_m3av.py \
      --train /data1/dohee/datasets/M3AV/manifests/m3av_math_train.csv \
      --dev /data1/dohee/datasets/M3AV/manifests/m3av_math_dev.csv \
      --train_clap_emb /data1/dohee/datasets/M3AV/clap_text_emb/m3av_math_train.pt \
      --save_dir results/ours/math_lam${tag} \
      --adapter_type gated \
      --pool_type mean \
      --align_loss_type cosine \
      --lambda_align ${lam} \
      --lambda_hidden 0.1 \
      --adapter_scale_init 0.01 \
      --lr 5e-5 \
      --epochs 20 \
      --batch_size 4 \
      --eval_batch_size 4 \
      --eval_num_beams 5 \
      > results/ours/math_lam${tag}.log 2>&1

    echo "Finished lambda=${lam}"
done
' > results/ours/math_two_lambda_runner.log 2>&1 &

# ours test
CUDA_VISIBLE_DEVICES=5 nohup bash -c '
for tag in 005 010; do
    echo "========================================"
    echo "Math Ours lam${tag} beam=5"
    echo "========================================"

    python -u eval_whisper_ours_m3av.py \
      --manifest /data1/dohee/datasets/M3AV/manifests/m3av_math_test.csv \
      --ckpt results/ours/math_lam${tag}/best.pt \
      --output_csv results/ours/math_lam${tag}/test_predictions_beam5.csv \
      --summary_json results/ours/math_lam${tag}/test_summary_beam5.json \
      --batch_size 16 \
      --num_beams 5 \
      --max_new_tokens 256
done
' > results/ours/math_test_beam5.log 2>&1 &