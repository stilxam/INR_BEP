from collections.abc import Sequence, Mapping
from typing import Optional
from secrets import randbelow

import jax
from jax import numpy as jnp
from jaxtyping import PyTree
import optax
import equinox as eqx

import common_jax_utils as cju
import common_dl_utils as cdu
from common_dl_utils.config_realization import PostponedInitialization


def complete_postponed_initialization(postponed_init:PostponedInitialization, completion: dict):
    postponed_init.resolve_missing_args(completion)
    for value in postponed_init.kwargs.values():
        if isinstance(value, PostponedInitialization):
            complete_postponed_initialization(value, completion)
        elif isinstance(value, Sequence):
            for v in value:
                if isinstance(v, PostponedInitialization):
                    complete_postponed_initialization(v, completion)

def run_single_experiment(missing_kwargs: dict[str, PyTree], incomplete_config: Mapping, key:jax.Array):
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
    return experiment.initialize()

def tree_unstack(tree, axis=0):
    leaves, tree_def = jax.tree.flatten(tree)
    array_leaf = next(filter(eqx.is_array, leaves))
    num_out = array_leaf.shape[axis]
    def _safe_unstack(maybe_array):
        if eqx.is_array(maybe_array):
            return jnp.unstack(maybe_array, axis=axis)
        else:
            return num_out*[maybe_array]
    unstacked_leaves = [_safe_unstack(leaf) for leaf in leaves]
    del leaves
    return [tree_def.unflatten(leaves) for leaves in zip(*unstacked_leaves)]

def run_parallel_experiments(
        missing_kwargs:dict[str, PyTree], 
        incomplete_config: Mapping,  
        post_processing_config: Mapping,
        key: Optional[jax.Array] = None,
        config_file_path: Optional[str] = None
        ):
    if key is None:
        key = jax.random.PRNGKey(seed=randbelow(2**32))
        print(f"Used key: {key}.")
    
    key_gen = cju.key_generator(key)

    post_processor = cdu.config_realization.get_model_from_config(  # we put this before the training
        config=post_processing_config,  # so that if there are any problems with the post_processing_config
        model_prompt = "post_processor_type",  # we find out before we spent time and resources training models
        default_module_key="components_module",
        initialize=False
    )
    post_processor = post_processor.initialize()

    # prepare the keys and 
    num_experiments = cdu.trees.get_first_leaf(missing_kwargs, is_leaf=eqx.is_array_like).shape[0]
    keys = jax.random.split(next(key_gen), num=num_experiments)
    results = eqx.filter_vmap(run_single_experiment, in_axes=(0, None, 0))(missing_kwargs, incomplete_config, keys)
    
    # process the results
    # this involves storing them and logging stuff to wandb, so we can't vmap
    results = tree_unstack(results)
    corresponding_parameters = tree_unstack(missing_kwargs)
    for result, experiment_parameters in zip(results, corresponding_parameters):
        post_processor(
            result, 
            experiment_parameters=experiment_parameters, 
            experiment_config=incomplete_config, 
            key=next(key_gen), 
            config_file_path=config_file_path
            )
        
