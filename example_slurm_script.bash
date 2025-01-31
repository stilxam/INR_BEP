#!/bin/bash
#SBATCH --account=tesr82932
#SBATCH --time=1:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=R-%x.%j.out
module load 2023
module load Miniconda3/23.5.2-0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash
conda activate snel_bep  # conda environment name

wandblogin="$(< ./wandb.login)"  # password stored in a file, don't add this file to your git repo!
wandb login "$wandblogin"


echo 'Starting new experiment!';
python run_parallel.py --config=./configs/example.yaml  # you can put more lines like this one after
# to do more groups of experements in sequence. 
# Snellius should be able to train a large batch of experiments in parallel in a very short time
# so it makes sense to do a couple of batches in sequence in the same script