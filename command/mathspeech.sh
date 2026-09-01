# whisper-base
CUDA_VISIBLE_DEVICES=3 python -u eval_whisper.py \
  --manifest /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/eval_format/mathspeech_test_source_seed42.csv \
  --model openai/whisper-base \
  --output_csv test_predictions_beam5.csv \
  --summary_json test_summary_beam5.json \
  --batch_size 16 \
  --num_workers 4 \
  --num_beams 5 \
  --max_new_tokens 256

# full fine-tuning train
CUDA_VISIBLE_DEVICES=3 nohup python -u train_whisper_fullft.py \
  --train /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/lora_format/mathspeech_train_source_seed42.csv \
  --dev /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/lora_format/mathspeech_val_source_seed42.csv \
  --output_dir results/fullft/mathspeech_source_disjoint \
  --model openai/whisper-base \
  --epochs 10 \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_workers 4 \
  --seed 42 \
  > results/fullft/mathspeech_source_disjoint/train.log 2>&1 &

# full fine-tuning test
CUDA_VISIBLE_DEVICES=3 python -u eval_whisper.py \
  --manifest /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/eval_format/mathspeech_test_source_seed42.csv \
  --model results/fullft/mathspeech_source_disjoint/best \
  --output_csv results/fullft/mathspeech_source_disjoint/test_predictions_beam5.csv \
  --summary_json results/fullft/mathspeech_source_disjoint/test_summary_beam5.json \
  --batch_size 16 \
  --num_workers 4 \
  --num_beams 5 \
  --max_new_tokens 256

# LoRA train
CUDA_VISIBLE_DEVICES=3 nohup python -u train_whisper_lora.py \
  --train /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/lora_format/mathspeech_train_source_seed42.csv \
  --dev /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/lora_format/mathspeech_val_source_seed42.csv \
  --output_dir results/lora/mathspeech_source_disjoint \
  --model openai/whisper-base \
  --rank 32 \
  --lora_alpha 32 \
  --epochs 10 \
  --learning_rate 1e-4 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_workers 4 \
  --seed 42 \
  > results/lora/mathspeech_source_disjoint/train.log 2>&1 &

# LoRA test
CUDA_VISIBLE_DEVICES=3 python -u eval_whisper.py \
  --manifest /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/eval_format/mathspeech_test_source_seed42.csv \
  --model results/lora/mathspeech_source_disjoint/merged \
  --output_csv results/lora/mathspeech_source_disjoint/test_predictions_beam5.csv \
  --summary_json results/lora/mathspeech_source_disjoint/test_summary_beam5.json \
  --batch_size 16 \
  --num_workers 4 \
  --num_beams 5 \
  --max_new_tokens 256



# ours train
CUDA_VISIBLE_DEVICES=3 nohup python -u train_mathspeech_projector_source_disjoint_hf.py \
  --train_csv /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/mathspeech_train_source_seed42.csv \
  --valid_csv /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/mathspeech_val_source_seed42.csv \
  --test_csv /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/mathspeech_test_source_seed42.csv \
  --save_dir results/projector/mathspeech/ours_final \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --whisper_name openai/whisper-base \
  --adapter_type gated \
  --pool_type cls \
  --adapter_bottleneck 256 \
  --dropout 0.1 \
  --adapter_scale_init 0.01 \
  --align_loss_type cosine \
  --lambda_align 0.05 \
  --lambda_hidden 0.1 \
  --batch_size 4 \
  --epochs 20 \
  --lr 5e-5 \
  --freeze_whisper \
  --seed 42 \
  --num_workers 4 \
  > results/projector/mathspeech/ours_final/train.log 2>&1 &

# ours eval (Whipser-normalized corpus-level jiwer WER/CER)
CUDA_VISIBLE_DEVICES=3 python -u eval_mathspeech_projector_source_disjoint_hf.py \
  --manifest /home/dohee/clap/Clap_Whipser_release/mathspeech/splits_source_balanced_seed42/mathspeech_test_source_seed42.csv \
  --ckpt train하고 저장된 ckpt경로 \
  --output_csv results/projector/mathspeech/ours_cls_lam010/test_predictions_beam5.csv \
  --summary_json results/projector/mathspeech/ours_cls_lam010/test_summary_beam5.json \
  --batch_size 16 \
  --num_beams 5 \
  --max_new_tokens 256 \
  --num_workers 2 \
  --fp16

# raw WER, CER은 compute_raw_asr_metrics.py 참고