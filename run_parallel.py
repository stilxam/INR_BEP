import traceback
import sys

import fire
import yaml
from jax import numpy as jnp

import common_dl_utils as cdu


from inr_utils.parallel_training import run_parallel_experiments


def main(
        config:str
        ):
    print(f"Running experiments from {config}")
    with open(config, 'r') as config_file:
        missing_kwargs, incomplete_config, post_processing_config = yaml.safe_load_all(config_file) # all three documents can be placed in the same file :)
    # make nice jax arrays out of the lists of values in missing_kwargs
    missing_kwargs = cdu.trees.tree_map(
        missing_kwargs,
        lambda x: jnp.asarray(x),
        is_leaf = lambda x: isinstance(x, (tuple, list))
    )
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