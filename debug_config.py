import traceback
import sys
import pdb

import fire
import yaml
from jax import numpy as jnp
import jax
import optax

import common_dl_utils as cdu
import common_jax_utils as cju

from inr_utils.parallel_training import complete_postponed_initialization, run_single_experiment


def main(
        config:str
        ):
    # Add these debug lines
    print("JAX devices:", jax.devices())
    print(f"Running experiments from {config}")
    with open(config, 'r') as config_file:
        missing_kwargs, incomplete_config, post_processing_config = yaml.safe_load_all(config_file) # all three documents can be placed in the same file :)
    # make nice jax arrays out of the lists of values in missing_kwargs
    missing_kwargs = cdu.trees.tree_map(
        missing_kwargs,
        lambda x: jnp.asarray(x)[0],
        is_leaf = lambda x: isinstance(x, (tuple, list))
    )
    key = jax.random.PRNGKey(0)
    post_processor = cdu.config_realization.get_model_from_config(
        config=post_processing_config,
        model_prompt="post_processor_type",
        default_module_key="components_module",
        initialize=False
    )
    print(post_processor)
    experiment = cju.run_utils.get_experiment_from_config_and_key(
        prng_key=key,
        config=incomplete_config,
        model_kwarg_in_trainer='inr',
        model_sub_config_name_base='model',  # so it looks for "model_config" in config
        trainer_default_module_key='trainer_module',  # so it knows to get the module specified by config.trainer_module
        additional_trainer_default_modules=[optax],  # remember the don't forget to add optax to the default modules? This is that 
        add_model_module_to_architecture_default_module=False,
        initialize=False  # don't run the experiment yet, we want to add the missing kwargs
    )
    complete_postponed_initialization(experiment, missing_kwargs)
    print(experiment)
    
    post_processor = post_processor.initialize()
    results = experiment.initialize()
    post_processor(results, experiment_parameters=missing_kwargs, experiment_config=incomplete_config, key=jax.random.PRNGKey(1), config_file_path=config)
    
    
    print("All finished")

if __name__ == '__main__':
    try:
        fire.Fire(main)
    except Exception as e:
        print(f"Got the following exception: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        pdb.post_mortem()
        raise e