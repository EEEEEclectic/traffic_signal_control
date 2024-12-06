#!/usr/bin/env bash

OUT_DIR="results_runs"
EPISODES=20
LR=0.001

# DCRNN Centralized
python training/train.py \
  --model_type dcrnn \
  --approach centralized \
  --episodes $EPISODES \
  --lr $LR \
  --exp_name "dcrnn_centralized" \
  --out_dir $OUT_DIR

# # DCRNN Decentralized
# python training/train.py \
#   --model_type dcrnn \
#   --approach decentralized \
#   --episodes $EPISODES \
#   --lr $LR \
#   --exp_name "dcrnn_decentralized" \
#   --out_dir $OUT_DIR

# # Transformer Centralized
# python training/train.py \
#   --model_type transformer \
#   --approach centralized \
#   --episodes $EPISODES \
#   --lr $LR \
#   --exp_name "transformer_centralized" \
#   --out_dir $OUT_DIR

# # Transformer Decentralized
# python training/train.py \
#   --model_type transformer \
#   --approach decentralized \
#   --episodes $EPISODES \
#   --lr $LR \
#   --exp_name "transformer_decentralized" \
#   --out_dir $OUT_DIR
