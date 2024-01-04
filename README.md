# slurm_example
Examples of how to run slurm jobs.

# run slurm job

```bash
sbatch run-test.sh
```

# Steps

The LAB machines with A40 GPUs: these all the necessary information on how to run experiments on our own GPUs. Here is a quick summary:

1. For convenience: add to your /etc/hosts on your local machine (mac/windows/linux etc.):

```
10.17.161.12    sprint1

10.17.161.13    sprint2
```

2. Login to the sprint1 machine: ssh your-user-name@sprint1

3. Add to the file /home/${USER}/.bashrc on the  sprint1 machine the following:
```
export PATH=$PATH:/usr/local/cuda-12.3/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.3/lib64

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sprint1/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	eval "$__conda_setup"
else
	if [ -f "/sprint1/anaconda/etc/profile.d/conda.sh" ]; then
    	. "/sprint1/anaconda/etc/profile.d/conda.sh"
	else
    	export PATH="/sprint1/anaconda/bin:$PATH"
	fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

4. Run the command: source /home/${USER}/.bashrc

5. Create a new environment:
```
conda create -n test-env
conda activate test-env
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```
6. Now you can run the slurm jobs on the sprint1 and sprint2 machines, check this minimal working example: https://github.com/sprintml/slurm_example 
