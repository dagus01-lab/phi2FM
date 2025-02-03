#!/bin/bash

source ~/miniconda3/bin/activate ddp_training

MASTER_ADDR=91.234.10.16
MASTER_PORT=12355
NODE_RANK=1
WORLD_SIZE=4

torchrun \
  --nnodes=3 \
  --nproc_per_node=1 \
  training_script.py \
  -r args/args_ddp_ccollado.yml \
  "$@"
