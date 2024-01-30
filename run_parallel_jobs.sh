#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --ntasks 16
#SBATCH --nodelist=sprint2

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

# activate the correct environment
# conda activate pytorch
# source activate /sprint1/anaconda/envs/pytorch
# source activate /home/${USER}/.conda/envs/test-env
conda activate test-env

# debug print-outs
echo USER: $USER
which conda
which python



echo "Job start at $(date)"

process_data() {
    # run the code
    PYTHONPATH=. python testslurm.py
}


export -f process_data
# if $1 is not empty then use it as the number of parallel jobs
if [ ! -z "$1" ]; then
  N_JOBS=$1
else
  N_JOBS=$SLURM_NTASKS
fi
echo $N_JOBS

parallel -linebuffer --delay .2 -j $N_JOBS --joblog "$(pwd)/joblog" --results "$(pwd)/output"

conda deactivate
echo "Job end at $(date)"
