# 2. Model Pretraining

1. Simply modify `args/phimultigpu_pretraining.yml` with your correct paths of input data and output files
2. Run `torchrun --nproc_per_node=4 training_script.py -r args/phimultigpu_pretraining.yml > pretrain.log 2>&1`
    - Runs pretraining with DDP on 1 node with 4 GPUs
    - Takes parameters from `args/phimultigpu_pretraining.yml`
    - Saves the output to `pretrain.log`