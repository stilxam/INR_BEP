# %%
import common_dl_utils as cdu
from common_dl_utils.config_creation import Config, VariableCollector
import json
import os
import numpy as np

# %%

config = Config()
variable = VariableCollector()

dataset_path = './DIV2K'
start_data_index = 0
end_data_index = 5


def put_all_data_in(dataset_path, start_data_index, end_data_index, top_level_name="entropy_class",
                    low_level_name="datapoint"):
    entropy_groups = []
    for entropy_group in os.listdir(dataset_path):
        group_files = os.listdir(f"{dataset_path}/{entropy_group}")[start_data_index:end_data_index]
        entropy_groups.append(variable(*[  # this assumes all entropy groups have the same number of files
            f"{dataset_path}/{entropy_group}/{file_name}"
            for file_name in group_files
        ], group=low_level_name))
    return variable(*entropy_groups, group=top_level_name)


from inr_utils.images import make_lin_grid

# param_pairs = np.linspace(8.33, 21.67, 4)
# param_pairs = np.array(param_pairs.reshape(-1, 2))
# w0s = param_pairs[:, 0]
# s0s = param_pairs[:, 1]


# first we specify what the model should look like
config.architecture = './model_components'  # module containing all relevant classes for architectures
# NB if the classes relevant for creating the model are spread over multiple modules, this is no problem
# let config.architecture be the module that contains the "main" model class, and for all other components just specify the module
# or specify the other modules as default modules to the tools in common_jax_utils.run_utils
config.model_type = 'inr_modules.CombinedINR'

config.model_config = Config()
config.get_ntk = True
config.model_config.in_size = 2
config.model_config.out_size = 3
config.model_config.terms = [
    # CombinedINR uses multiple MLPs and returns the sum of their outputs. These 'terms' are the MLPs
    ('inr_modules.MLPINR.from_config', {
        'hidden_size': 256,
        'num_layers': 5,
        'layer_type': variable(
            'inr_layers.SirenLayer',
            'inr_layers.SinCardLayer',
            'inr_layers.HoscLayer',
            'inr_layers.AdaHoscLayer',
            'inr_layers.RealWIRE',
            'inr_layers.GaussianINRLayer',
            'inr_layers.QuadraticLayer',
            'inr_layers.MultiQuadraticLayer',
            'inr_layers.LaplacianLayer',
            'inr_layers.ExpSinLayer',
            'inr_layers.FinerLayer',
            group='method'),
        'num_splits': 3,
        'activation_kwargs': variable(
            {'w0': variable(*tuple(np.linspace(18.33, 31.67, 4)), group="hyperparam")},
            # SIREN #  you can nest variables and put complex datastructures in their slots
            {'w0': variable(*tuple(np.linspace(18.33, 31.67, 4)), group="hyperparam")},  # SinCard
            {'w0': variable(*tuple(np.linspace(3.67, 10.33, 4)), group="hyperparam")},  # HOSC
            {'w0': variable(*tuple(np.linspace(3.33, 12.67, 4)), group="hyperparam")},  # AdaHOSC
            {'w0': variable(*tuple(np.linspace(8.33, 21.67, 4)), group="hyperparam"),
             's0': variable(*tuple(np.linspace(8.33, 21.67, 4)), group="hyperparam")},  # WIRE
            {'inverse_scale': variable(*tuple(np.linspace(9.17, 13.83, 4)), group="hyperparam")},  # Gaussian
            {'a': variable(*tuple(np.linspace(29.17, 45.83, 4)), group="hyperparam")},  # Quadratic
            {'a': variable(*tuple(np.linspace(33.33, 46.67, 4)), group="hyperparam")},  # MultiQuadratic
            {'a': variable(*tuple(np.linspace(0.292, 1.258, 4)), group="hyperparam")},  # Laplacian
            # {'a': variable(*tuple(np.linspace(0.03, 1., 6)), group="hyperparam"), 'b': variable(*tuple(np.linspace(1.0, 2., 6)), group="hyperparam")}, #SuperGaussian
            {'a': variable(*tuple(np.linspace(2.5, 8.5, 4)), group="hyperparam")},  # ExpSin
            {'w0': variable(*tuple(np.linspace(17.5, 27.5, 4)), group="hyperparam")},  # Finer
            group='method'),
        # by specifying a group, you make sure that the values in the group are linked, so SirenLayer wil always go with w0 and GaussianINRLayer with inverse_scale
    }),
]

# next, we set up the training loop, including the 'target_function' that we want to mimic
config.trainer_module = './inr_utils/'
config.trainer_type = 'training.train_inr_scan'  # 'training.train_inr'  # NB you can use a different training loop, e.g. training.train_inr_scan instead to make it train much faster
config.loss_evaluator = 'losses.PointWiseLossEvaluator'
config.target_function = 'images.ContinuousImage'
config.target_function_config = {
    'image': put_all_data_in(dataset_path, start_data_index, end_data_index),
    'scale_to_01': False,
    'interpolation_method': 'images.make_piece_wise_constant_interpolation',
    'minimal_coordinate': -1.,
    'maximal_coordinate': 1.,
}
config.data_index = None
config.loss_function = 'losses.scaled_mse_loss'
config.take_grad_of_target_function = False
# config.state_update_function = ('auxiliary.ilm_updater', {num_steps = 10000})
# config.state_update_function = ('state_test_objects.py', 'counter_updater')
config.sampler = ('sampling.GridSubsetSampler', {
    # samples coordinates in a fixed grid, that should in this case coincide with the pixel locations in the image
    'size': [1356, 2040],  # [240, 320], group="datapoint"),#[2040, 1356],
    'batch_size': 27120,  # 32*240, group="datapoint"),#2000,
    'allow_duplicates': False,
    'min': -1.
})

config.optimizer = 'adam'  # we'll have to add optax to the additional default modules later
# config.optimizer = 'sgd'
config.optimizer_config = {
    'learning_rate': 1.e-4
}
config.steps = 100000  # changed from 40000
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

# config.after_training_callback = None  # don't care for one now, but you could have this e.g. store some nice loss plots if you're not using wandb
config.optimizer_state = None  # we're starting from scratch

config.components_module = "./inr_utils/"
config.post_processor_type = "post_processing.PostProcessor"
config.storage_directory = variable(
    "factory_results/Siren_image",
    "factory_results/SinCard_image",
    "factory_results/Hosc_image",
    "factory_results/AdaHosc_image",
    "factory_results/WIRE_image",
    "factory_results/Gaussian_image",
    "factory_results/Quadratic_image",
    "factory_results/MultiQuadratic_image",
    "factory_results/Laplacian_image",
    "factory_results/ExpSin_image",
    "factory_results/Finer_image", group="method")
config.wandb_kwargs = {}
config.metrics = [
    ('metrics.MSEOnFixedGrid', {
        'grid': [1356, 2040],  # [240, 320], group="datapoint"),
        'batch_size': 8160,  # 2400, group="datapoint"),
        'frequency': 'every_n_batches'
    }),
]

config.entropy_class = variable(1, 2, 3, 4, 5, group="entropy_class")
config.method_name = variable(
    "Siren_image",
    'SinCard_image',
    'Hosc_image',
    'AdaHosc_image',
    'WIRE_image',
    'Gaussian_image',
    'Quadratic_image',
    'MultiQuadratic_image',
    'Laplacian_image',
    'ExpSin_image',
    'Finer_image', group="method")

# config.wandb_group = variable("Siren_image", 'SinCard_image', 'Hosc_image', 'AdaHosc_image', 'WIRE_image', 'Gaussian_image', 'Quadratic_image', 'MultiQuadratic_image', 'Laplacian_image', 'ExpSin_image', 'Finer_image', group="method")
config.wandb_entity = "INR_NTK"
config.wandb_project = "images"
# %%
target_dir = "./factory_configs/images"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

config_files = []


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Config):
            return o.data
        return super().default(o)


for config_index, config_realization in enumerate(variable.realizations(config)):
    group = f"{config_realization['entropy_class']}-{config_realization['method_name']}"
    config_realization["wandb_group"] = group
    target_path = f"{target_dir}/{group}-{config_index}.yaml"
    with open(target_path, "w") as yaml_file:
        json.dump(config_realization, yaml_file, cls=MyEncoder)
    config_files.append(target_path)

# %%
variable._group_to_lengths
# %%
len(list(variable.realizations(config)))
# %%
# now create a slurm file that does what we want  NB you'll need to modify th account probably
# and the time
slurm_directory = "./factory_slurm/images"
if not os.path.exists(slurm_directory):
    os.makedirs(slurm_directory)
# chnage account and output directory and maybe conda env name
slurm_base = """#!/bin/bash
#SBATCH --account=tesr82931  
#SBATCH --time=0:20:00
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

# conda init bash
conda activate inr_edu_24  # conda environment name


echo 'Starting new experiment!';
"""

for config_file in config_files:
    slurm_script = slurm_base + f"\npython run_single.py --config={config_file}"
    slurm_file_name = (config_file.split("/")[-1].split(".")[0]) + ".bash"
    with open(f"{slurm_directory}/{slurm_file_name}", "w") as slurm_file:
        slurm_file.write(slurm_script)
