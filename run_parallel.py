import traceback
import sys

import fire
import yaml
from jax import numpy as jnp

import common_dl_utils as cdu


from inr_utils.parallel_training import run_parallel_experiments

def main(config: str):
    print(f"Running experiments from {config}")
    with open(config, 'r') as config_file:
        missing_kwargs, incomplete_config, post_processing_config = yaml.safe_load_all(config_file)
    
    # More careful conversion of missing_kwargs to arrays
    def convert_to_array(x):
        if isinstance(x, (tuple, list)):
            return jnp.asarray(x)
        return x

    # Convert only the leaf nodes that are lists/tuples
    missing_kwargs = cdu.trees.tree_map(
        convert_to_array,  # Pass the function directly, not the result
        missing_kwargs,
        is_leaf=lambda x: isinstance(x, (tuple, list))
    )
    
    # Ensure all values in activation_kwargs are arrays
    if 'activation_kwargs' in missing_kwargs:
        missing_kwargs['activation_kwargs'] = {
            k: jnp.asarray(v) if isinstance(v, (list, tuple)) else v
            for k, v in missing_kwargs['activation_kwargs'].items()
        }
    
    run_parallel_experiments(
        missing_kwargs=missing_kwargs,
        incomplete_config=incomplete_config,
        post_processing_config=post_processing_config,
        config_file_path=config
    )
    print("All finished")

if __name__ == '__main__':
    try:
        fire.Fire(main)
    except Exception as e:
        print(f"Got the following exception: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e