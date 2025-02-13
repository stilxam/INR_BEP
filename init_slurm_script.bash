#!/bin/bash
#SBATCH --account=tesr82932
#SBATCH --time=1:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=./slurm_output/R-%x.%j.out
module load 2023
module load Miniconda3/23.5.2-0
#source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh

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

#if using audio files, run this first
# python convert_audio_to_npy.py

#Initializing scale factor experiments: 

echo 'Starting scale factor experiments!';
python run_parallel.py --config=./configs/InitialScheme/example_siren.yaml
python run_parallel.py --config=./configs/InitialScheme/finer_scheme.yaml
python run_parallel.py --config=./configs/InitialScheme/standard_scheme.yaml
