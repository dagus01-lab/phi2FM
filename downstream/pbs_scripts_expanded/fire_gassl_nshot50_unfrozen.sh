#!/bin/bash
#PBS -P 1001
#PBS -N FR_gass_n50_f
#PBS -q gpu4_dbg
#PBS -l walltime=00:05:00
#PBS -l select=1:ngpus=4:ncpus=96:mem=739g

source /lustre/projects/1001/miniconda3/bin/activate
conda activate esa-phisatnet

cd /lustre/projects/1001/gdaga/home/phi2FM/downstream

python training_script.py -r "args/lustre_expanded/fire/gassl_nshot50_unfrozen.yml"
