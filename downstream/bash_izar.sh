#!/bin/bash -l
#SBATCH --job-name=izar_pretrain
#SBATCH --nodes=1               # one node only
#SBATCH --gres=gpu:2            # 2 GPUs on this node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --output=izar_%j.log    # stdout/stderr combined

# 1) Conda
source /home/ccollado/miniconda3/etc/profile.d/conda.sh
conda activate esa-phisatnet

# 2) Move to project root
cd /home/ccollado/phi2FM/downstream

# 3) OpenMP threads for dataloading, etc.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4) Launch training
python training_script.py -r args/izar/phisatnet.yml
