#%%
import common_dl_utils as cdu
from common_dl_utils.config_creation import Config, VariableCollector
import json
import os
import numpy as np
#%%
config = Config()
variable = VariableCollector()


config = Config()

# first we specify what the model should look like
config.architecture = './model_components'  # module containing all relevant classes for architectures
# NB if the classes relevant for creating the model are spread over multiple modules, this is no problem
# let config.architecture be the module that contains the "main" model class, and for all other components just specify the module
# or specify the other modules as default modules to the tools in common_jax_utils.run_utils
config.model_type = 'inr_modules.CombinedINR'

config.model_config = Config()
config.model_config.in_size = 1
config.model_config.out_size = 1
config.model_config.terms = [  # CombinedINR uses multiple MLPs and returns the sum of their outputs. These 'terms' are the MLPs
    ('inr_modules.MLPINR.from_config',{
        'hidden_size': 256,
        'num_layers': 5,
        'layer_type': variable('inr_layers.SirenLayer', 'inr_layers.SinCardLayer', 'inr_layers.HoscLayer', 'inr_layers.AdaHoscLayer', 'inr_layers.RealWIRE', 'inr_layers.GaussianINRLayer', 'inr_layers.QuadraticLayer', 'inr_layers.MultiQuadraticLayer', 'inr_layers.LaplacianLayer', 'inr_layers.SuperGaussianLayer', 'inr_layers.ExpSinLayer', 'inr_layers.FinerLayer', group='method'),
        'num_splits': 3,
        'activation_kwargs': variable(
            {'w0':variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #SIREN #  you can nest variables and put complex datastructures in their slots
            {'w0':variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #SinCard
            {'w0': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #HOSC
            {'w0': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #AdaHOSC
            {'w0': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam"), 's0': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #WIRE
            {'inverse_scale':variable(*np.linspace(1., 50., 2), group="hyperparam")}, #Gaussian
            {'a': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #Quadratic
            {'a': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #MultiQuadratic
            {'a': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #Laplacian
            {'a': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam"), 'b': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #SuperGaussian
            {'a': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #ExpSin
            {'w0': variable(*tuple(np.linspace(1., 50., 2)), group="hyperparam")}, #Finer
            group='method'),  # by specifying a group, you make sure that the values in the group are linked, so SirenLayer wil always go with w0 and GaussianINRLayer with inverse_scale
    }),
]

# next, we set up the training loop, including the 'target_function' that we want to mimic
config.trainer_module = './inr_utils/'  
config.trainer_type = 'training.train_inr_scan'#'training.train_inr'  # NB you can use a different training loop, e.g. training.train_inr_scan instead to make it train much faster
config.loss_evaluator = 'losses.SoundLossEvaluator'
config.loss_function = 'losses.SoundLossEvaluator'
config.loss_evaluator_config = {
    'time_domain_weight': 1.0,
    'frequency_domain_weight': 0.000001
}

# config.target_function = 'images.ContinuousImage'
# config.target_function_config = {
#     'image': variable('./example_data/Audio/Audio_entropy_5p_25p/hiphop.00080.wav', "example_data/Audio/Audio_entropy_bottom_5p/rock.00086.wav", group="datapoint"),#'./example_data/gray_parrot_grads_scaled.npy',
#     'scale_to_01': False,
#     'interpolation_method': 'images.make_piece_wise_constant_interpolation',
#     'minimal_coordinate': -1.,
#     'maximal_coordinate':1.,
# }   
config.sampler = ('sampling.SoundSampler', {
    'window_size': 256,
    'batch_size': 32,
    'allow_pickle': True,
    'fragment_length': None,
    'sound_fragment': "./example_data/data_gt_bach.npy",
})

config.data_index = None
# config.loss_function = 'losses.scaled_mse_loss'
# config.take_grad_of_target_function = False
#config.state_update_function = ('auxiliary.ilm_updater', {num_steps = 10000})
# config.state_update_function = ('state_test_objects.py', 'counter_updater')
# config.sampler = ('sampling.GridSubsetSampler',{  # samples coordinates in a fixed grid, that should in this case coincide with the pixel locations in the image
#     'size': variable([2040, 1356], [240, 320], group="datapoint"),#[2040, 1356],
#     'batch_size': variable(27120, 32*240, group="datapoint"),#2000,
#     'allow_duplicates': False,
#     'min':-1.
# })

config.optimizer = 'adam'  # we'll have to add optax to the additional default modules later
# config.optimizer = 'sgd'
config.optimizer_config = {
    'learning_rate': 1.e-4,
    'b1': 0.8,
    'b2': 0.999999
}
config.steps = 160000 #changed from 40000
# config.use_wandb = True

# # now we want some extra things, like logging, to happen during training
# # the inr_utils.training.train_inr function allows for this through callbacks.
# # The callbacks we want to use can be found in inr_utils.callbacks
# config.after_step_callback = 'callbacks.ComposedCallback'
# config.after_step_callback_config = {
#     'callbacks':[
#         ('callbacks.print_loss', {'after_every':400}),  # only print the loss every 400th step
#         'callbacks.report_loss',  # but log the loss to wandb after every step
#         ('callbacks.MetricCollectingCallback', # this thing will help us collect metrics and log images to wandb
#              {'metric_collector':'metrics.MetricCollector'}
#         ),
#         'callbacks.raise_error_on_nan'  # stop training if the loss becomes NaN
#     ],
#     'show_logs': False
# }

# config.after_training_callback = ('state_test_objects.py', 'after_training_callback')

# config.metric_collector_config = {  # the metrics for MetricCollectingCallback / metrics.MetricCollector
#     'metrics':[
#         # ('metrics.PlotOnGrid2D', {'grid': 256, 'batch_size':8*256, 'frequency':'every_n_batches'}),  
#         # # ^ plots the image on this fixed grid so we can visually inspect the inr on wandb
#         # ('metrics.MSEOnFixedGrid', {'grid': [2040, 1356], 'batch_size':2040, 'frequency': 'every_n_batches'})
#         # ^ compute the MSE with the actual image pixels
#         ('metrics.ImageGradMetrics', {
#             'grid':variable([2040, 1356],[240, 320], group="datapoint"), 
#             'batch_size': variable(2040, 2400, group="datapoint"), 
#             'frequency': 'every_n_batches'
#             }),
#     ],
#     'batch_frequency': 400,  # compute all of these metrics every 400 batches
#     'epoch_frequency': 1  # not actually used
# }

grsz = np.load("./example_data/data_gt_bach.npy").shape[0]

#config.after_training_callback = None  # don't care for one now, but you could have this e.g. store some nice loss plots if you're not using wandb 
config.optimizer_state = None  # we're starting from scratch


config.components_module = "./inr_utils/"
config.post_processor_type = "post_processing.PostProcessor"
config.storage_directory = variable("factory_results/Siren_audio_grad", "factory_results/SinCard_audio_grad", "factory_results/Hosc_audio_grad", "factory_results/AdaHosc_audio_grad", "factory_results/WIRE_audio_grad", "factory_results/Gaussian_audio_grad", "factory_results/Quadratic_audio_grad", "factory_results/MultiQuadratic_audio_grad", "factory_results/Laplacian_audio_grad", "factory_results/SuperGaussian_audio_grad", "factory_results/ExpSin_audio_grad", "factory_results/Finer_audio_grad", group="method")
config.wandb_kwargs = {}
config.metrics = [
    {
    'metrics': [
        ('metrics.AudioMetricsOnGrid', {
            'target_audio': "./example_data/data_gt_bach.npy",
            'grid_size': grsz,
            'batch_size': 1024,  # This will be automatically adjusted if needed
            'sr': 16000,
            'frequency': 'every_n_batches',
            'save_path': variable("factory_results/Siren_audio_grad", "factory_results/SinCard_audio_grad", "factory_results/Hosc_audio_grad", "factory_results/AdaHosc_audio_grad", "factory_results/WIRE_audio_grad", "factory_results/Gaussian_audio_grad", "factory_results/Quadratic_audio_grad", "factory_results/MultiQuadratic_audio_grad", "factory_results/Laplacian_audio_grad", "factory_results/SuperGaussian_audio_grad", "factory_results/ExpSin_audio_grad", "factory_results/Finer_audio_grad", group="method")
        })
    ],
    'batch_frequency': 100,
    'epoch_frequency': 1
}
]

config.wandb_group = variable("Siren_audio_grad", 'SinCard_audio_grad', 'Hosc_audio_grad', 'AdaHosc_audio_grad', 'WIRE_audio_grad', 'Gaussian_audio_grad', 'Quadratic_audio_grad', 'MultiQuadratic_audio_grad', 'Laplacian_audio_grad', 'SuperGaussian_audio_grad', 'ExpSin_audio_grad', 'Finer_audio_grad', group="method")
config.wandb_entity = "INR_NTK"
config.wandb_project = "audi_reconstruction"



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
