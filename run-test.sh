#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --job-name=test_single_gpu
#SBATCH --ntasks 16
#SBATCH --nodelist=sprint2

# mitigates activation problems
eval "$(conda shell.bash hook)"

# activate the correct environment
# conda activate pytorch
source activate /sprint1/anaconda/envs/pytorch

# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python testslurm.py
