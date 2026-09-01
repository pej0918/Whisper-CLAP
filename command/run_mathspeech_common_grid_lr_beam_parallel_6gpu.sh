#!/usr/bin/env bash
set -euo pipefail

# MathSpeech 6-GPU grid using the teammate MathSpeech pipeline.
# GPUS="0 1 2 3 4 5" bash command/run_mathspeech_common_grid_lr_beam_parallel_6gpu.sh

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
OURS_BATCH_SIZE="${OURS_BATCH_SIZE:-4}"
CLAP_FULLFT_BATCH_SIZE="${CLAP_FULLFT_BATCH_SIZE:-4}"
RESIDUAL_BATCH_SIZE="${RESIDUAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GPUS_STR="${GPUS:-0 1 2 3 4 5}"
read -r -a GPU_LIST <<< "${GPUS_STR}"
BEAMS_STR="${BEAMS:-1 5}"
read -r -a BEAMS <<< "${BEAMS_STR}"

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
# FullFT / LoRA / Residual use teammate segmented-manifest format.
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/lora_format/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/lora_format/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/eval_format/mathspeech_test_source_seed42.csv}"
# Ours and CLAP-guided Full FT use the source-disjoint sample_id/reference_text manifests.
OURS_TRAIN_CSV="${OURS_TRAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
OURS_VAL_CSV="${OURS_VAL_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
OURS_TEST_CSV="${OURS_TEST_CSV:-${SPLIT_DIR}/mathspeech_test_source_seed42.csv}"
CLAP_EMB="${CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"

OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_seed${SEED}_6gpu}"
LORA_TARGETS_WITH_OUTPROJ="${LORA_TARGETS_WITH_OUTPROJ:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"
LORA_TARGETS_NO_OUTPROJ="${LORA_TARGETS_NO_OUTPROJ:-q_proj,k_proj,v_proj,fc1,fc2}"
mkdir -p "${OUT_ROOT}/logs"

require_file(){ [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
for f in "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}" "${OURS_TRAIN_CSV}" "${OURS_VAL_CSV}" "${OURS_TEST_CSV}" "${CLAP_EMB}"; do require_file "$f"; done

echo "[CONFIG] OUT_ROOT=${OUT_ROOT} GPUS=${GPU_LIST[*]} EPOCHS=${EPOCHS} BEAMS=${BEAMS[*]}"
echo "Full/LoRA train=${TRAIN_CSV}"
echo "Ours/CLAP-FullFT train=${OURS_TRAIN_CSV}"
echo "LoRA with out_proj=${LORA_TARGETS_WITH_OUTPROJ}"
echo "LoRA no out_proj=${LORA_TARGETS_NO_OUTPROJ}"

run_eval_whisper(){
  local gpu="$1" model_path="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_whisper.py \
    --manifest "${TEST_CSV}" --model "$model_path" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --num_beams "$beam" --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_eval_projector(){
  local gpu="$1" ckpt="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_mathspeech_projector_source_disjoint_hf.py \
    --manifest "${OURS_TEST_CSV}" --ckpt "$ckpt" \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --num_beams "$beam" --max_new_tokens "${MAX_NEW_TOKENS}" --fp16 \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_eval_residual(){
  local gpu="$1" ckpt="$2" out_dir="$3" beam="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_whisper_residual_adapter.py \
    --manifest "${TEST_CSV}" --ckpt "$ckpt" --model "${MODEL}" --adapter_bottleneck 256 \
    --output_csv "${out_dir}/test_predictions_beam${beam}.csv" \
    --summary_json "${out_dir}/test_summary_beam${beam}.json" \
    --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --num_beams "$beam" --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${out_dir}/eval_beam${beam}.log"
}

run_fullft_job(){
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/fullft_lr${lr}"; mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_whisper_fullft_compat.py \
    --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "$out_dir" --model "${MODEL}" \
    --epochs "${EPOCHS}" --learning_rate "$lr" --train_batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
    --generation_num_beams 5 --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_whisper "$gpu" "${out_dir}/best" "$out_dir" "$beam"; done
}

run_ours_job(){
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/ours_lr${lr}"; mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_mathspeech_projector_source_disjoint_hf.py \
    --train_csv "${OURS_TRAIN_CSV}" --valid_csv "${OURS_VAL_CSV}" --test_csv "${OURS_TEST_CSV}" \
    --save_dir "$out_dir" --clap_emb_path "${CLAP_EMB}" --whisper_name "${MODEL}" \
    --adapter_type gated --pool_type cls --adapter_bottleneck 256 --dropout 0.1 --adapter_scale_init 0.01 \
    --align_loss_type cosine --lambda_align 0.05 --lambda_hidden 0.1 \
    --batch_size "${OURS_BATCH_SIZE}" --epochs "${EPOCHS}" --lr "$lr" --weight_decay 1e-4 \
    --freeze_whisper --seed "${SEED}" --num_workers "${NUM_WORKERS}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_projector "$gpu" "${out_dir}/best.pt" "$out_dir" "$beam"; done
}

run_clap_fullft_job(){
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/clap_fullft_lr${lr}"; mkdir -p "$out_dir"
  echo "[CLAP-GUIDED FULL FT] lr=${lr}; Whisper trainable"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_mathspeech_projector_source_disjoint_hf.py \
    --train_csv "${OURS_TRAIN_CSV}" --valid_csv "${OURS_VAL_CSV}" --test_csv "${OURS_TEST_CSV}" \
    --save_dir "$out_dir" --clap_emb_path "${CLAP_EMB}" --whisper_name "${MODEL}" \
    --adapter_type gated --pool_type cls --adapter_bottleneck 256 --dropout 0.1 --adapter_scale_init 0.01 \
    --align_loss_type cosine --lambda_align 0.05 --lambda_hidden 0.1 \
    --batch_size "${CLAP_FULLFT_BATCH_SIZE}" --epochs "${EPOCHS}" --lr "$lr" --weight_decay 1e-4 \
    --seed "${SEED}" --num_workers "${NUM_WORKERS}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_projector "$gpu" "${out_dir}/best.pt" "$out_dir" "$beam"; done
}

run_lora_job(){
  local gpu="$1" lr="$2" variant="$3" targets="$4"
  local out_dir="${OUT_ROOT}/lora_whisper_lr${lr}_${variant}"; mkdir -p "$out_dir"
  echo "[LORA] variant=${variant} targets=${targets} lr=${lr}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_whisper_lora_controlled.py \
    --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --output_dir "$out_dir" --model "${MODEL}" \
    --rank 32 --lora_alpha 32 --target_modules "${targets}" \
    --epochs "${EPOCHS}" --learning_rate "$lr" --train_batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" --gradient_accumulation_steps 1 --num_workers "${NUM_WORKERS}" \
    --generation_num_beams 5 --generation_max_length "${MAX_NEW_TOKENS}" --seed "${SEED}" \
    2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_whisper "$gpu" "${out_dir}/merged" "$out_dir" "$beam"; done
}

run_residual_job(){
  local gpu="$1" lr="$2" out_dir="${OUT_ROOT}/residual_b256_lr${lr}"; mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_whisper_residual_adapter.py \
    --train "${TRAIN_CSV}" --dev "${VAL_CSV}" --save_dir "$out_dir" --model "${MODEL}" \
    --epochs "${EPOCHS}" --lr "$lr" --batch_size "${RESIDUAL_BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --adapter_bottleneck 256 --selection_num_beams 5 --selection_max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" --fp16 2>&1 | tee "${out_dir}/train.log"
  for beam in "${BEAMS[@]}"; do run_eval_residual "$gpu" "${out_dir}/best.pt" "$out_dir" "$beam"; done
}

# Base: no training, beam 1 and 5.
mkdir -p "${OUT_ROOT}/whisper_base"
for beam in "${BEAMS[@]}"; do run_eval_whisper "${GPU_LIST[0]}" "${MODEL}" "${OUT_ROOT}/whisper_base" "$beam"; done

pids=(); names=()
launch(){ local name="$1"; shift; echo "[LAUNCH] $name"; ( "$@" ) > "${OUT_ROOT}/logs/${name}.launcher.log" 2>&1 & pids+=("$!"); names+=("$name"); }

# Wave 1: six jobs, one per GPU.
launch "fullft_lr1e-5_gpu${GPU_LIST[0]}" run_fullft_job "${GPU_LIST[0]}" 1e-5
launch "fullft_lr1e-4_gpu${GPU_LIST[1]}" run_fullft_job "${GPU_LIST[1]}" 1e-4
launch "ours_lr1e-5_gpu${GPU_LIST[2]}" run_ours_job "${GPU_LIST[2]}" 1e-5
launch "ours_lr1e-4_gpu${GPU_LIST[3]}" run_ours_job "${GPU_LIST[3]}" 1e-4
launch "lora_outproj_lr1e-5_gpu${GPU_LIST[4]}" run_lora_job "${GPU_LIST[4]}" 1e-5 outproj "${LORA_TARGETS_WITH_OUTPROJ}"
launch "lora_outproj_lr1e-4_gpu${GPU_LIST[5]}" run_lora_job "${GPU_LIST[5]}" 1e-4 outproj "${LORA_TARGETS_WITH_OUTPROJ}"
status=0
for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "[DONE] ${names[$i]}"; else echo "[FAILED] ${names[$i]}" >&2; status=1; fi; done
[[ "$status" -eq 0 ]] || { echo "[ERROR] First wave failed" >&2; exit "$status"; }

# Wave 2: all six GPUs used again.
pids=(); names=()
launch "residual_lr1e-5_gpu${GPU_LIST[0]}" run_residual_job "${GPU_LIST[0]}" 1e-5
launch "residual_lr1e-4_gpu${GPU_LIST[1]}" run_residual_job "${GPU_LIST[1]}" 1e-4
launch "lora_nooutproj_lr1e-5_gpu${GPU_LIST[2]}" run_lora_job "${GPU_LIST[2]}" 1e-5 nooutproj "${LORA_TARGETS_NO_OUTPROJ}"
launch "lora_nooutproj_lr1e-4_gpu${GPU_LIST[3]}" run_lora_job "${GPU_LIST[3]}" 1e-4 nooutproj "${LORA_TARGETS_NO_OUTPROJ}"
launch "clap_fullft_lr1e-5_gpu${GPU_LIST[4]}" run_clap_fullft_job "${GPU_LIST[4]}" 1e-5
launch "clap_fullft_lr1e-4_gpu${GPU_LIST[5]}" run_clap_fullft_job "${GPU_LIST[5]}" 1e-4
status=0
for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "[DONE] ${names[$i]}"; else echo "[FAILED] ${names[$i]}" >&2; status=1; fi; done
[[ "$status" -eq 0 ]] || { echo "[ERROR] Second wave failed" >&2; exit "$status"; }

"${PYTHON}" -u scripts/collect_asr_results.py --output_dir "${OUT_ROOT}"
echo "[DONE] ${OUT_ROOT}"
