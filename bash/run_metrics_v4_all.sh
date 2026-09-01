#!/bin/bash

BASE=/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments
OUT=$BASE/metrics_summary_v4_fixedgen48_clean.csv

echo "method,csv,pred_col,num_eval,WER,CER,MathTermRecall,OverBiasRate,TailRate,AvgLenRatio" > $OUT

for EXP in \
v4_ce_residual_mlp_hidden \
v4_ce_gated_hidden \
v4_ce_bottleneck_hidden \
v4_align_residual_mlp_cosine_hidden \
v4_align_residual_mlp_clip_hidden \
v4_align_residual_mlp_cosine_clip_hidden \
v4_align_gated_attn_cosine_hidden
do
  CSV=$BASE/$EXP/result_ASR_test_fixedgen48_clean.csv
  METRIC=$BASE/$EXP/metrics_test_fixedgen48_clean.csv

  echo "=============================="
  echo "Computing metrics for $EXP"
  echo "CSV: $CSV"
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
