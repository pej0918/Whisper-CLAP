# whisper-base
CUDA_VISIBLE_DEVICES=<빈GPU> python -u eval_whisper.py \
  --model openai/whisper-base \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_cs_test.csv \
  --output_csv results/base/cs_test_predictions_beam5.csv \
  --summary_json results/base/cs_test_summary_beam5.json \
  --batch_size 16 \
  --num_beams 5 \
  --max_new_tokens 256

# full fine-tuning train
CUDA_VISIBLE_DEVICES=3 nohup python -u train_whisper_fullft.py \
  --train /data1/dohee/datasets/M3AV/manifests/m3av_cs_train.csv \
  --dev /data1/dohee/datasets/M3AV/manifests/m3av_cs_dev.csv \
  --output_dir results/full_ft/cs \
  --epochs 3 \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  > results/full_ft/cs_train.log 2>&1 &


# full fine-tuning test
CUDA_VISIBLE_DEVICES=3 python eval_whisper.py \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_cs_test.csv \
  --output_csv results/full_ft/cs_test.csv \
  --summary_json results/full_ft/cs_test.json \
  --model results/full_ft/cs/best \
  --batch_size 16 \
  --num_beams 5

# LoRA train
CUDA_VISIBLE_DEVICES=4 nohup python -u train_whisper_lora.py \
  --train /data1/dohee/datasets/M3AV/manifests/m3av_cs_train.csv \
  --dev /data1/dohee/datasets/M3AV/manifests/m3av_cs_dev.csv \
  --output_dir results/lora/cs \
  --epochs 10 \
  --learning_rate 1e-4 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  > results/lora/cs_train.log 2>&1 &

# LoRA test
CUDA_VISIBLE_DEVICES=3 python eval_whisper.py \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_cs_test.csv \
  --output_csv results/lora/cs_test.csv \
  --summary_json results/lora/cs_test.json \
  --model results/lora/cs/merged \
  --batch_size 16 \
  --num_beams 5


# ours train
CUDA_VISIBLE_DEVICES=4 nohup python -u train_whisper_ours_m3av.py \
  --train /data1/dohee/datasets/M3AV/manifests/m3av_cs_train.csv \
  --dev /data1/dohee/datasets/M3AV/manifests/m3av_cs_dev.csv \
  --train_clap_emb /data1/dohee/datasets/M3AV/clap_text_emb/m3av_cs_train.pt \
  --save_dir results/ours/cs_lam010 \
  --adapter_type gated \
  --pool_type mean \
  --align_loss_type cosine \
  --lambda_align 0.10 \
  --lambda_hidden 0.1 \
  --adapter_scale_init 0.01 \
  --lr 5e-5 \
  --epochs 20 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --eval_num_beams 5 \
  > results/ours/cs_lam010.log 2>&1 &