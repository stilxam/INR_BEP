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
    print(f"Used key: {key}.")

    wandb_group = config_dict["wandb_group"]
    wandb_entity = config_dict["wandb_entity"]
    wandb_project = config_dict["wandb_project"]
    with wandb.init(config=config_dict, group=wandb_group, project=wandb_project, entity=wandb_entity):
        experiment = cju.run_utils.get_experiment_from_config_and_key(
            prng_key=key,
            config=config_dict,
            model_kwarg_in_trainer='inr',
            model_sub_config_name_base='model',  # so it looks for "model_config" in config
            trainer_default_module_key='trainer_module',  # so it knows to get the module specified by config.trainer_module
            additional_trainer_default_modules=[optax],  # remember the don't forget to add optax to the default modules? This is that 
            add_model_module_to_architecture_default_module=False,
            initialize=False  # don't run the experiment yet, we want to use wandb
        )
        print(experiment)
        experiment.initialize()
        print("Finished")

if __name__ == '__main__':
    try:
        fire.Fire(main)
    except Exception as e:
        print(f"Got the following exception: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e