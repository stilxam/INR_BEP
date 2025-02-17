import traceback
import sys
from secrets import randbelow
from typing import Optional

import fire
import yaml
import jax
import optax
import wandb

import common_jax_utils as cju
import common_dl_utils as cdu
import jax.numpy as jnp


from ntk.sweep import make_init_apply
from ntk.analysis import get_NTK_ntvp, decompose_ntk, measure_of_diagonal_strength
from ntk.visualization import plot_ntk_kernels
from inr_utils.images import make_lin_grid
from matplotlib import pyplot as plt


def flatten_kernel(K):
    return K.transpose([0, 2, 1, 3]).reshape([K.shape[0] * K.shape[2], K.shape[1] * K.shape[3]])

def main(
        config:str,
        seed:Optional[int]=None
        ):
    print(f"Running experiments from {config}")
    with open(config, 'r') as config_file:
        config_dict = yaml.safe_load(config_file) # just a single config

    if seed is None:
        seed=randbelow(2**32)
    print(f"Used seed: {seed}")
    key = jax.random.PRNGKey(seed=seed)
    key_gen = cju.key_generator(key)
    print(f"Used key: {key}.")

    if config_dict.get("post_processor_type", None) is not None:
        post_processor = cju.run_utils.get_model_from_config_and_key(#cdu.config_realization.get_model_from_config(  # we put this before the training
            prng_key=next(key_gen),
            add_model_module_to_architecture_default_module=False,
            config=config_dict,  # so that if there are any problems with the post_processing_config
            model_prompt = "post_processor_type",  # we find out before we spent time and resources training models
            default_module_key="components_module",
            initialize=False
        )
        post_processor = post_processor.initialize()
    else:
        next(key_gen)  # just so that the rest experiment gets the same keys regardless of whether we have a post processor
        post_processor = None

    wandb_group = config_dict["wandb_group"]
    wandb_entity = config_dict["wandb_entity"]
    wandb_project = config_dict["wandb_project"]
    with wandb.init(
        config=cdu.config_creation.make_flat_config(config_dict), 
        group=wandb_group, 
        project=wandb_project, 
        entity=wandb_entity
        ):
        init_key = next(key_gen)
        if config_dict.get("compute_ntk", False):
            init_fn, apply_fn, inr = make_init_apply(config_dict, init_key)
            params = init_fn()
            in_dims = config_dict["model_config"]["in_size"]
            grid_size = int(100**(1/in_dims))
            locations = make_lin_grid(0, 1, grid_size, in_dims)
            flat_locations = locations.reshape(-1, in_dims)
            ntvp = get_NTK_ntvp(apply_fn)
            NTK = ntvp(flat_locations, flat_locations, params)

            if NTK.ndim > 2:
                NTK = flatten_kernel(NTK)
            _, _, _, condition_number = decompose_ntk(NTK)
            lin_measure = measure_of_diagonal_strength(NTK, map_kwarg=0)

            ntk_vis = plot_ntk_kernels(NTK, config_dict["model_config"]["terms"][0][1]["layer_type"], config_dict["model_config"]["terms"][0][1]["activation_kwargs"])

            wandb.log({
                "ntk_condition_number": jnp.log(condition_number + 1e-5),
                "lin_measure": float(lin_measure),
                "ntk_plot": wandb.Image(ntk_vis),
            })

        experiment = cju.run_utils.get_experiment_from_config_and_key(
            prng_key=init_key,
            config=config_dict,
            model_kwarg_in_trainer='inr',
            model_sub_config_name_base='model',  # so it looks for "model_config" in config
            trainer_default_module_key='trainer_module',  # so it knows to get the module specified by config.trainer_module
            additional_trainer_default_modules=[optax],  # remember the don't forget to add optax to the default modules? This is that 
            add_model_module_to_architecture_default_module=False,
            initialize=False  # don't run the experiment yet, we want to use wandb
        )
        print(experiment)
        result = experiment.initialize()
        print("Finished training")
        if post_processor is not None:
            post_processor(
            result, 
            experiment_parameters={}, 
            experiment_config=config_dict, 
            key=next(key_gen), 
            config_file_path=config
            )
            print("Finished post processing")

if __name__ == '__main__':
    try:
        fire.Fire(main)
    except Exception as e:
        print(f"Got the following exception: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e