#!/bin/bash
#SBATCH --account=my_snellius_account
#SBATCH --time=2:00:00
#SBATCH -p gpu_mig
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=R-%x.%j.out
module load 2022
module load Miniconda3/4.12.0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL8/EB_production/2022/software/Miniconda3/4.12.0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash
conda activate inr_edu_24  # conda environment name

wandblogin="$(< ../wandb.login)"  # password stored in a file, don't add this file to your git repo!
wandb login "$wandblogin"


echo 'Starting new experiment!';
python run_from_inr_sweep.py --sweep_id=e8varw5l  # appropriate sweep id from wandb sweep