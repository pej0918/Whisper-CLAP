#!/usr/bin/env bash
set -euo pipefail

# MathSpeech lambda search for Full Ours.
# - 18 lambda candidates
# - GPU 0..5, 3 concurrent jobs per GPU
# - fixed LR=1e-4, epochs=10, batch=4, beam=5 validation selection
# - NO test evaluation during lambda selection
# - after selecting the best Full lambda by validation WER, retrain CE-only,
#   CE+Hidden, CE+Align with the selected weights and evaluate all 4 variants
#   on the MathSpeech test set with beam=1 and beam=5.
#
# Usage:
#   bash command/run_mathspeech_lambda_search_6gpu.sh
# Optional:
#   GPUS="0 1 2 3 4 5" bash command/run_mathspeech_lambda_search_6gpu.sh

ROOT="${ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-openai/whisper-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SELECTION_BEAM="${SELECTION_BEAM:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GPUS_STR="${GPUS:-0 1 2 3 4 5}"
read -r -a GPU_LIST <<< "${GPUS_STR}"

if [[ ${#GPU_LIST[@]} -ne 6 ]]; then
  echo "[ERROR] Expected exactly 6 GPUs, got: ${GPU_LIST[*]}" >&2
  exit 1
fi

SPLIT_DIR="${SPLIT_DIR:-${ROOT}/mathspeech/splits_source_balanced_seed42}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/mathspeech_train_source_seed42.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/mathspeech_val_source_seed42.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/mathspeech_test_source_seed42.csv}"
CLAP_EMB="${CLAP_EMB:-/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt}"

OUT_ROOT="${OUT_ROOT:-/data1/eunju/clap_whisper_results/mathspeech_lambda_search_seed${SEED}}"
SEARCH_ROOT="${OUT_ROOT}/full_search"
FINAL_ROOT="${OUT_ROOT}/final_ablation"
mkdir -p "${SEARCH_ROOT}" "${FINAL_ROOT}" "${OUT_ROOT}/logs"

# Existing current Full (.05,.10) run can be included as an anchor candidate.
EXISTING_FULL_DIR="${EXISTING_FULL_DIR:-/data1/eunju/clap_whisper_results/mathspeech_common_grid_bestvalidwer_seed42_6gpu/ours_lr1e-4}"

require_file(){ [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
for f in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$CLAP_EMB"; do require_file "$f"; done

run_train(){
  local gpu="$1" lam_a="$2" lam_h="$3" out_dir="$4" align_type="$5"
  mkdir -p "$out_dir"
  echo "[START] gpu=$gpu lambda_align=$lam_a lambda_hidden=$lam_h out=$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/train_mathspeech_projector_best_wer.py \
    --train_csv "$TRAIN_CSV" \
    --valid_csv "$VAL_CSV" \
    --test_csv "$TEST_CSV" \
    --save_dir "$out_dir" \
    --clap_emb_path "$CLAP_EMB" \
    --whisper_name "$MODEL" \
    --adapter_type gated \
    --pool_type cls \
    --adapter_bottleneck 256 \
    --dropout 0.1 \
    --adapter_scale_init 0.01 \
    --align_loss_type "$align_type" \
    --lambda_align "$lam_a" \
    --lambda_hidden "$lam_h" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay 1e-4 \
    --selection_num_beams "$SELECTION_BEAM" \
    --selection_max_new_tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --freeze_whisper \
    > "$out_dir/train.log" 2>&1
  echo "[DONE] gpu=$gpu lambda_align=$lam_a lambda_hidden=$lam_h"
}

# Candidate list: align hidden
CANDIDATES=(
  "0.025 0.001"
  "0.025 0.005"
  "0.025 0.010"
  "0.050 0.001"
  "0.050 0.005"
  "0.050 0.010"
  "0.050 0.025"
  "0.050 0.050"
  "0.075 0.001"
  "0.075 0.005"
  "0.075 0.010"
  "0.075 0.025"
  "0.075 0.050"
  "0.100 0.001"
  "0.100 0.005"
  "0.100 0.010"
  "0.100 0.025"
  "0.150 0.010"
)

pids=()
names=()

for idx in "${!CANDIDATES[@]}"; do
  read -r lam_a lam_h <<< "${CANDIDATES[$idx]}"
  gpu_idx=$((idx / 3))
  gpu="${GPU_LIST[$gpu_idx]}"
  tag="a${lam_a}_h${lam_h}"
  out_dir="${SEARCH_ROOT}/${tag}"
  echo "[LAUNCH] $tag -> GPU $gpu"
  ( run_train "$gpu" "$lam_a" "$lam_h" "$out_dir" cosine ) \
    > "${OUT_ROOT}/logs/${tag}_gpu${gpu}.launcher.log" 2>&1 &
  pids+=("$!")
  names+=("$tag")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[DONE] ${names[$i]}"
  else
    echo "[FAILED] ${names[$i]}" >&2
    status=1
  fi
done
[[ "$status" -eq 0 ]] || { echo "[ERROR] At least one lambda-search job failed." >&2; exit "$status"; }

SELECT_ARGS=(
  --search_root "$SEARCH_ROOT"
  --output_json "$OUT_ROOT/selected_lambda.json"
  --output_env "$OUT_ROOT/selected_lambda.env"
  --ranking_csv "$OUT_ROOT/lambda_ranking.csv"
)
if [[ -f "$EXISTING_FULL_DIR/training_summary.json" && -f "$EXISTING_FULL_DIR/best.pt" ]]; then
  SELECT_ARGS+=(--extra_dir "$EXISTING_FULL_DIR")
fi

"$PYTHON" -u scripts/select_mathspeech_lambda.py "${SELECT_ARGS[@]}" \
  2>&1 | tee "$OUT_ROOT/lambda_selection.log"

# shellcheck disable=SC1090
source "$OUT_ROOT/selected_lambda.env"

echo "[SELECTED] lambda_align=$SELECTED_ALIGN lambda_hidden=$SELECTED_HIDDEN"
echo "[SELECTED] ckpt=$SELECTED_CKPT"

# Final ablation uses the selected Full weights.
# Full itself is NOT retrained: the validation-selected search checkpoint is final.
# CE-only, CE+Hidden, CE+Align are retrained with identical architecture/protocol.

FINAL_CE_ONLY="$FINAL_ROOT/ce_only"
FINAL_CE_HIDDEN="$FINAL_ROOT/ce_hidden"
FINAL_CE_ALIGN="$FINAL_ROOT/ce_align"
FINAL_FULL="$FINAL_ROOT/full_ours"
mkdir -p "$FINAL_CE_ONLY" "$FINAL_CE_HIDDEN" "$FINAL_CE_ALIGN" "$FINAL_FULL"

# Symlink selected Full outputs into final_ablation for a clean final table.
ln -sfn "$SELECTED_DIR/best.pt" "$FINAL_FULL/best.pt"
ln -sfn "$SELECTED_DIR/train_log.csv" "$FINAL_FULL/train_log.csv"
ln -sfn "$SELECTED_DIR/training_summary.json" "$FINAL_FULL/training_summary.json"

pids=(); names=()
( run_train "${GPU_LIST[0]}" 0.0 0.0 "$FINAL_CE_ONLY" none ) > "$OUT_ROOT/logs/final_ce_only.launcher.log" 2>&1 &
pids+=("$!"); names+=("ce_only")
( run_train "${GPU_LIST[1]}" 0.0 "$SELECTED_HIDDEN" "$FINAL_CE_HIDDEN" none ) > "$OUT_ROOT/logs/final_ce_hidden.launcher.log" 2>&1 &
pids+=("$!"); names+=("ce_hidden")
( run_train "${GPU_LIST[2]}" "$SELECTED_ALIGN" 0.0 "$FINAL_CE_ALIGN" cosine ) > "$OUT_ROOT/logs/final_ce_align.launcher.log" 2>&1 &
pids+=("$!"); names+=("ce_align")

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[DONE] final ${names[$i]}"; else echo "[FAILED] final ${names[$i]}" >&2; status=1; fi
done
[[ "$status" -eq 0 ]] || { echo "[ERROR] Final ablation training failed." >&2; exit "$status"; }

run_eval(){
  local gpu="$1" name="$2" ckpt="$3" beam="$4"
  local out_dir="$FINAL_ROOT/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_mathspeech_projector_source_disjoint_hf.py \
    --manifest "$TEST_CSV" \
    --ckpt "$ckpt" \
    --output_csv "$out_dir/test_predictions_beam${beam}.csv" \
    --summary_json "$out_dir/test_summary_beam${beam}.json" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_beams "$beam" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --fp16 \
    > "$out_dir/eval_beam${beam}.log" 2>&1
}

# Final test happens only after lambda selection is frozen.
pids=(); names=()
variants=("ce_only" "ce_hidden" "ce_align" "full_ours")
ckpts=(
  "$FINAL_CE_ONLY/best.pt"
  "$FINAL_CE_HIDDEN/best.pt"
  "$FINAL_CE_ALIGN/best.pt"
  "$SELECTED_CKPT"
)

for i in "${!variants[@]}"; do
  gpu="${GPU_LIST[$i]}"
  name="${variants[$i]}"
  ckpt="${ckpts[$i]}"
  (
    run_eval "$gpu" "$name" "$ckpt" 1
    run_eval "$gpu" "$name" "$ckpt" 5
  ) > "$OUT_ROOT/logs/final_eval_${name}_gpu${gpu}.launcher.log" 2>&1 &
  pids+=("$!"); names+=("$name")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[DONE] final eval ${names[$i]}"; else echo "[FAILED] final eval ${names[$i]}" >&2; status=1; fi
done
[[ "$status" -eq 0 ]] || { echo "[ERROR] Final test evaluation failed." >&2; exit "$status"; }

"$PYTHON" - "$FINAL_ROOT" "$OUT_ROOT/selected_lambda.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sel_path = Path(sys.argv[2])
with open(sel_path) as f:
    selected = json.load(f)["selected"]

names = [
    ("CE only", "ce_only"),
    ("CE + Hidden", "ce_hidden"),
    ("CE + Align", "ce_align"),
    ("Full Ours", "full_ours"),
]

print("=" * 86)
print("FINAL MATHSPEECH RESULTS")
print("=" * 86)
print(f"Selected lambda_align  = {selected['lambda_align']}")
print(f"Selected lambda_hidden = {selected['lambda_hidden']}")
print(f"Selected valid WER     = {selected['best_valid_wer']:.6f}")
print("-" * 86)
print(f"{'Method':<18} {'WER b1':>12} {'CER b1':>12} {'WER b5':>12} {'CER b5':>12}")
print("-" * 86)
rows = []
for label, d in names:
    vals = {}
    for beam in (1, 5):
        p = root / d / f"test_summary_beam{beam}.json"
        with open(p) as f:
            x = json.load(f)
        vals[beam] = (float(x['wer']), float(x['cer']))
    print(f"{label:<18} {vals[1][0]:>12.6f} {vals[1][1]:>12.6f} {vals[5][0]:>12.6f} {vals[5][1]:>12.6f}")
    rows.append({
        'method': label,
        'wer_beam1': vals[1][0],
        'cer_beam1': vals[1][1],
        'wer_beam5': vals[5][0],
        'cer_beam5': vals[5][1],
    })
print("=" * 86)

with open(root / "final_mathspeech_results.json", "w") as f:
    json.dump({
        'selected_lambda_align': selected['lambda_align'],
        'selected_lambda_hidden': selected['lambda_hidden'],
        'selected_valid_wer': selected['best_valid_wer'],
        'results': rows,
    }, f, indent=2)
PY

echo "[DONE] Lambda search + final MathSpeech ablation/test complete."
echo "[RESULT] $FINAL_ROOT/final_mathspeech_results.json"
