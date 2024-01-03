#!/bin/bash

#SBATCH --gres=gpu:3
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --ntasks 16
#SBATCH --job-name=test_multigpu
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

echo $CUDA_VISIBLE_DEVICES

# run the code
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=3 test_multigpu.py
