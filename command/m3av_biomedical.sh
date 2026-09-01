# whisper-base
CUDA_VISIBLE_DEVICES=<빈GPU> python -u eval_whisper.py \
  --model openai/whisper-base \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_test.csv \
  --output_csv results/base/biomedical_test_predictions_beam5.csv \
  --summary_json results/base/biomedical_test_summary_beam5.json \
  --batch_size 16 \
  --num_beams 5 \
  --max_new_tokens 256

# full fine-tuning train
CUDA_VISIBLE_DEVICES=3 nohup python -u train_whisper_fullft.py \
  --train /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_train.csv \
  --dev /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_dev.csv \
  --output_dir results/full_ft/biomedical \
  --epochs 3 \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  > results/full_ft/biomedical_train.log 2>&1 &


# full fine-tuning test
CUDA_VISIBLE_DEVICES=3 python eval_whisper.py \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_test.csv \
  --output_csv results/full_ft/biomedical_test.csv \
  --summary_json results/full_ft/biomedical_test.json \
  --model results/full_ft/biomedical/best \
  --batch_size 16 \
  --num_beams 5

# LoRA train 
CUDA_VISIBLE_DEVICES=4 nohup python -u train_whisper_lora.py \
  --train /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_train.csv \
  --dev /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_dev.csv \
  --output_dir results/lora/biomedical \
  --epochs 10 \
  --learning_rate 1e-4 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  > results/lora/biomedical_train.log 2>&1 &

# LoRA test
CUDA_VISIBLE_DEVICES=3 python eval_whisper.py \
  --manifest /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_test.csv \
  --output_csv results/lora/biomedical_test.csv \
  --summary_json results/lora/biomedical_test.json \
  --model results/lora/biomedical/merged \
  --batch_size 16 \
  --num_beams 5

# ours train
for cfg in "005 0.05" "010 0.10"; do
    set -- $cfg
    tag=$1
    lam=$2

    echo "========================================"
    echo "Biomedical Ours lambda=${lam}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=5 python -u train_whisper_ours_m3av.py \
      --train /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_train.csv \
      --dev /data1/dohee/datasets/M3AV/manifests/m3av_biomedical_dev.csv \
      --train_clap_emb /data1/dohee/datasets/M3AV/clap_text_emb/m3av_biomedical_train.pt \
      --save_dir results/ours/biomedical_lam${tag} \
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
      > results/ours/biomedical_lam${tag}.log 2>&1

    echo "Finished Biomedical lambda=${lam}"
done

# ours test