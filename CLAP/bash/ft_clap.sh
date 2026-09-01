#!/bin/bash
# rm -rf /data1/eunju/outputs/clap_mlp_spoken/mlp_clap_spoken
 
CUDA_VISIBLE_DEVICES=2 python -m laion_clap.training.main \
  --dataset-type webdataset \
  --datasetpath /data1/eunju/datasets/mlp_clap_spoken \
  --datasetnames "webdataset" \
  --datasetinfos "train" \
  --resume /data1/eunju/model_ckpt/CLAP/630k-fusion-best.pt \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --enable-fusion \
  --fusion-type "aff_2d" \
  --logs /data1/eunju/outputs/clap_mlp_spoken \
  --name mlp_clap_spoken_epoch5 \
  --batch-size 8 \
  --epochs 5 \
  --workers 2 \
  --precision fp32 \
  --save-frequency 1 \
  --save-most-recent \
  --no-eval \
  --data-filling repeatpad \
  --data-truncating fusion \