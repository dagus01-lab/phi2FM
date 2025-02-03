#!/bin/bash

source ~/miniconda3/bin/activate base

MASTER_ADDR=91.234.10.16
MASTER_PORT=12355
NODE_RANK=0
WORLD_SIZE=4

torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  training_script.py \
  -r args/args_ddp_phimultigpu.yml \
  "$@"
