#!/bin/bash
#PBS -N FR_unip_n5000_f
#PBS -q gpu4_std
#PBS -l walltime=23:00:00
#PBS -l select=1:ngpus=4:ncpus=30:mem=200g

source /lustre/projects/1001/miniconda3/bin/activate
conda activate esa-phisatnet

cd /lustre/projects/1001/gdaga/home/phi2FM/downstream
export PYTHONPATH=$PYTHONPATH:/lustre/projects/1001/gdaga/home/phi2FM
python training_script.py -r "args/lustre_expanded/fire/uniphi_nshot5000_unfrozen.yml"
