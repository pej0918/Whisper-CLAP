#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# rm -rf /data1/eunju/outputs/clap_mlp_spoken/mlp_clap_spoken_eval

CUDA_VISIBLE_DEVICES=2 python -m laion_clap.training.main \
  --dataset-type webdataset \
  --datasetpath /data1/eunju/datasets/mlp_clap_spoken \
  --datasetnames webdataset \
  --datasetinfos train \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --enable-fusion \
  --fusion-type aff_2d \
  --data-filling repeatpad \
  --data-truncating fusion \
  --logs /data1/eunju/outputs/clap_mlp_spoken \
  --batch-size 8 \
  --epochs 0 \
  --workers 2 \
  --precision fp32 \
  --resume /data1/eunju/model_ckpt/CLAP/630k-fusion-best.pt \
  --name mlp_clap_spoken_eval_base

#    --resume /data1/eunju/outputs/clap_mlp_spoken/mlp_clap_spoken/checkpoints/epoch_1.pt \ 
#    --name mlp_clap_spoken_eval \