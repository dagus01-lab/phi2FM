#!/bin/bash

###############################################
# central_launcher.sh
# Usage: bash central_launcher.sh [additional args...]
###############################################

# 1. Set global environment variables
export MASTER_ADDR="91.234.10.16"
export MASTER_PORT="29500"
NNODES=3
WORLD_SIZE=5

# 2. Define any arguments you might pass through
#    This will forward extra CLI arguments to the torchrun commands
EXTRA_ARGS="$@"

############################
# 3. LAUNCH ON LOCAL NODE (phimultigpu, node_rank=0)
############################

# Activate the Conda environment for the master node
echo "[phimultigpu] Activating conda environment 'base'"
source ~/miniconda3/bin/activate base

echo "[phimultigpu] Starting local training with nproc_per_node=2 ..."
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node=2 \
  --node_rank=0 \
  training_script.py \
  -r args/args_ddp_phimultigpu.yml \
  ${EXTRA_ARGS} > out_phimultigpu.log 2>&1 &

PHIMULTIGPU_PID=$!

############################
# 4. LAUNCH ON CCOLLADO (node_rank=1)
############################

echo "[ccollado] Starting remote training ..."
ssh ccollado "bash -s" <<EOF > out_ccollado.log 2>&1 &
echo "Debug: This runs on ccollado before activating conda."
# Activate the Conda environment on the remote machine
source ~/miniconda3/bin/activate ddp_training
echo "Debug: This runs after conda activation. Now launching torchrun..."

# Run torchrun with the appropriate rank, nproc_per_node, etc.
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=1 \
  --node_rank=1 \
  training_script.py \
  -r args/args_ddp_ccollado.yml \
  ${EXTRA_ARGS}
EOF

CCOLLADO_PID=$!

############################
# 5. LAUNCH ON PHISAT2 (node_rank=2)
############################

echo "[phisat2] Starting remote training ..."
ssh phisat2 "bash -s" <<EOF > out_phisat2.log 2>&1 &
echo "Debug: This runs on ccollado before activating conda."
# Activate the Conda environment on the remote machine
source ~/miniconda3/bin/activate ddp_training
echo "Debug: This runs after conda activation. Now launching torchrun..."

# Run torchrun with the appropriate rank
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=1 \
  --node_rank=2 \
  training_script.py \
  -r args/args_ddp_phisat2.yml \
  ${EXTRA_ARGS}
EOF

PHISAT2_PID=$!

############################
# 6. WAIT FOR ALL PROCESSES
############################

wait ${PHIMULTIGPU_PID}
wait ${CCOLLADO_PID}
wait ${PHISAT2_PID}

echo "All training processes have completed."
