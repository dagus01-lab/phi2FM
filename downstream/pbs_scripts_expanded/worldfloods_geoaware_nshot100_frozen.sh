#!/bin/bash
#PBS -N WF_geoa_n100_f
#PBS -q gpu4_std
#PBS -l walltime=00:05:00
#PBS -l select=1:ngpus=4:ncpus=30:mem=200g

source /lustre/projects/1001/miniconda3/bin/activate
conda activate esa-phisatnet

cd /lustre/projects/1001/gdaga/home/phi2FM/downstream
export PYTHONPATH=$PYTHONPATH:/lustre/projects/1001/gdaga/home/phi2FM
python training_script.py -r "args/lustre_expanded/worldfloods/geoaware_nshot100_frozen.yml"
