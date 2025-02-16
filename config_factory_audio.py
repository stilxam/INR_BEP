#%%
import common_dl_utils as cdu
from common_dl_utils.config_creation import Config, VariableCollector
import json
import os
import numpy as np
#%%
def create_audio_configs():
    config = Config()
    variable = VariableCollector()

    # Model architecture configuration
    config.architecture = './model_components'
    config.model_type = 'inr_modules.CombinedINR'
    
    config.model_config = Config()
    config.model_config.in_size = 1
    config.model_config.out_size = 1
    config.model_config.terms = [
        ('inr_modules.MLPINR.from_config', {
            'hidden_size': 256,
            'num_layers': 5,
            'layer_type': 'inr_layers.SirenLayer',
            'num_splits': 1,
            'use_complex': False,
            'initialization_scheme': 'initialization_schemes.siren_scheme',
            'activation_kwargs': variable(
                {'w0': variable(*tuple(np.linspace(15., 50., 8)), group="hyperparam")},
            ),
            'positional_encoding_layer': ['inr_layers.ClassicalPositionalEncoding.from_config', {
                'num_frequencies': 10
            }]
        })]

    # Training configuration
    config.trainer_module = './inr_utils/'
    config.trainer_type = 'training.train_inr_scan'
    
    config.loss_evaluator = 'losses.SoundLossEvaluator'
    config.loss_function = 'losses.SoundLossEvaluator'
    config.loss_evaluator_config = {
        'time_domain_weight': 1.0,
        'frequency_domain_weight': 0.000001
    }

    # Sampler configuration
    config.sampler = ('sampling.SoundSampler', {
        'window_size': 256,
        'batch_size': 32,
        'allow_pickle': True,
        'fragment_length': None,
        'sound_fragment': './example_data/data_gt_bach.npy'
    })

    # Optimizer configuration
    config.optimizer = 'adam'
    config.optimizer_config = {
        'b1': 0.8,
        'b2': 0.999999,
        'learning_rate': variable(*[1.e-4, 1.5e-4, 1.e-4, 1.5e-4, 1.e-4, 1.5e-4, 1.e-4, 1.5e-4], group="hyperparam")
    }
    config.steps = 20000

    # Post-processing configuration
    config.components_module = './inr_utils/'
    config.post_processor_type = 'post_processing.PostProcessor'
    config.metrics = [
        ('metrics.AudioMetricsOnGrid', {
            'target_audio': './example_data/data_gt_bach.npy',
            'grid_size': None,
            'batch_size': 1024,
            'sr': 16000,
            'frequency': 'every_n_batches',
            'save_path': variable(
                './results/audio_test/reconstructed_w0_{w0}_lr_{lr}.wav',
                group="hyperparam"
            )
        })
    ]

    config.batch_frequency = 1
    config.storage_directory = variable(
        './results/audio_test/w0_{w0}_lr_{lr}',
        group="hyperparam"
    )
    
    config.wandb_kwargs = {
        'project': 'inr_edu_24',
        'group': 'audio_test'
    }

    return config, variable

def main():
    config, variable = create_audio_configs()

    # Create config files
    target_dir = "./factory_configs/audio_test"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    config_files = []

    class MyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, Config):
                return o.data
            return super().default(o)

    # Generate configurations
    for config_index, config_realization in enumerate(variable.realizations(config)):
        # Format the save paths with actual values
        w0 = config_realization['model_config']['terms'][0][1]['activation_kwargs']['w0']
        lr = config_realization['optimizer_config']['learning_rate']
        
        config_realization['metrics'][0][1]['save_path'] = \
            config_realization['metrics'][0][1]['save_path'].format(w0=w0, lr=lr)
        config_realization['storage_directory'] = \
            config_realization['storage_directory'].format(w0=w0, lr=lr)

        target_path = f"{target_dir}/audio_config_{config_index}.yaml"
        with open(target_path, "w") as yaml_file:
            json.dump(config_realization, yaml_file, cls=MyEncoder)
        config_files.append(target_path)

    # Create SLURM scripts
    slurm_directory = "./factory_slurm/audio_test"
    if not os.path.exists(slurm_directory):
        os.makedirs(slurm_directory)

    slurm_base = """#!/bin/bash
#SBATCH --account=tesr82932
#SBATCH --time=2:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=./factory_output/R-%x.%j.out

module load 2023
module load Miniconda3/23.5.2-0

# >>> conda initialize >>>
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

conda init bash
conda activate snel_bep

wandblogin="$(< ./wandb.login)"
wandb login "$wandblogin"

echo 'Starting new audio experiment!';
"""

    for config_file in config_files:
        slurm_script = slurm_base + f"\npython run_single.py --config={config_file}"
        slurm_file_name = (config_file.split("/")[-1].split(".")[0]) + ".bash"
        with open(f"{slurm_directory}/{slurm_file_name}", "w") as slurm_file:
            slurm_file.write(slurm_script)

if __name__ == "__main__":
    main()

#%%
variable._group_to_lengths
#%%
len(list(variable.realizations(config)))
#%%
target_dir = "./factory_configs/test"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

config_files = []

class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Config):
            return o.data
        return super().default(o)

for config_index, config_realization in enumerate(variable.realizations(config)):
    group = config_realization["wandb_group"]
    target_path = f"{target_dir}/{group}-{config_index}.yaml"
    with open(target_path, "w") as yaml_file:
        json.dump(config_realization, yaml_file, cls=MyEncoder)
    config_files.append(target_path)
    
#%%
# now create a slurm file that does what we want  NB you'll need to modify th account probably
# and the time
slurm_directory = "./factory_slurm/test"
if not os.path.exists(slurm_directory):
    os.makedirs(slurm_directory)
#chnage account and output directory and maybe conda env name
slurm_base = """#!/bin/bash
#SBATCH --account=tesr82932
#SBATCH --time=0:15:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=./factory_output/R-%x.%j.out
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
"""

for config_file in config_files:
    slurm_script = slurm_base + f"\npython run_single.py --config={config_file}"
    slurm_file_name = (config_file.split("/")[-1].split(".")[0])+".bash"
    with open(f"{slurm_directory}/{slurm_file_name}", "w") as slurm_file:
        slurm_file.write(slurm_script)

#%%
