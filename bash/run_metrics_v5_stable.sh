#!/bin/bash

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
OUT=$BASE/metrics_summary_v5_stable_fixedgen48_clean.csv

echo "method,csv,pred_col,num_eval,WER,CER,MathTermRecall,OverBiasRate,TailRate,AvgLenRatio" > $OUT

for EXP in \
v5_ce_residual_mlp_hidden_scale0001 \
v5_ce_gated_hidden_scale0001 \
v5_align_residual_mlp_cosine_lam003_hidden05_scale0001 \
v5_align_residual_mlp_cosine_lam001_hidden05_scale0001 \
v5_ce_residual_mlp_hidden1_scale0001 \
v5_align_residual_mlp_cosine_lam003_hidden1_scale0001
do
  CSV=$BASE/$EXP/result_ASR_test_fixedgen48_clean.csv
  METRIC=$BASE/$EXP/metrics_test_fixedgen48_clean.csv

  echo "=============================="
  echo "Computing metrics for $EXP"
  echo "=============================="

  python compute_asr_metrics_v4.py \
    --csv $CSV \
    --out_csv $METRIC

  python - << EOF >> $OUT
import pandas as pd
m = pd.read_csv("$METRIC").iloc[0].to_dict()
print(f"$EXP,{m['csv']},{m['pred_col']},{m['num_eval']},{m['WER']},{m['CER']},{m['MathTermRecall']},{m['OverBiasRate']},{m['TailRate']},{m['AvgLenRatio']}")
EOF

done

echo "===================================="
echo "Saved summary to: $OUT"
echo "===================================="
cat $OUT
