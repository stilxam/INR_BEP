""" 
Script for running an experiment from a hyperparameter sweep. Specifically for hypernetwork sweeps.
Due to annoying issues with wandb, it's best to set count to 1 and just repeatedly call this script.
"""
import fire  # this allows us to change a function into a command line interface
from common_jax_utils.wandb_utils import run_from_wandb
import wandb
from functools import partial
import optax
import project_utils
import traceback
import sys

def main(  # since we run this using fire, all parameters become command line arguments
        sweep_id:str,
        count:int,
        project:str='crystals',
        entity:str='nld',
        model_kwarg_in_trainer:str='hypernetwork',
        model_sub_config_name_base:str='model',
        trainer_default_module_key:str='trainer_module',
        add_model_module_to_architecture_default_module:bool=False,
        trainer_activation_method:str='train',
        provide_prng_key_to_trainer_activation_method:bool=True,
        **kwargs
        ):
    """ 
    Start a wandb agent and run count experiments from a hyperparameter sweep.

    :parameter sweep_id: the id of the wandb sweep
    :parameter count: the number of experiments to run with the same wandb agent
        NB due to a problem with wandb, this is best kept to 1. Just call this script multiple times using a bash script.
    :parameter project: the wandb project
    :parameter entity: the wandb entity
    :parameter model_kwarg_in_trainer: the key in the trainer that should be used to pass the model.
        in the trainer class from hypernetwork_utils, this is 'hypernetwork'
    :parameter model_sub_config_name_base: the base name for the subconfig that should be used for the model.
        The full name will be model_sub_config_name_base + '_config'
        So by default, the we start looking in model_config for the (hyper) parameters for creating the model.
    :parameter trainer_default_module_key: the key in the config pointing to the module that should be used as the default module for the trainer.
        Typically in our project, this should be 'trainer_module', and config['trainer_module'] should then be './hypernetwork_utils'
    :parameter add_model_module_to_architecture_default_module: whether the model module should be added to the default module for the architecture.
        This is useful when the model class is defined in a different module from the layers and components that make up the architecture.
        In our project, this should typically be false since model_components contains both the model class and the various layers and submodels used for the architecture.
    :parameter trainer_activation_method: the method that should be called on the trainer to start training.
    :parameter provide_prng_key_to_trainer_activation_method: whether a prng key should be provided to the trainer activation method.
    :parameter kwargs: additional keyword arguments that should be passed to the common_jax_utils.run_from_wandb function.
    """
    wandb.login()
    full_sweep_id = f"{entity}/{project}/{sweep_id}"
    print(f"Starting {count} runs for {full_sweep_id}")
    wandb.agent(
        sweep_id=full_sweep_id,
        function=partial(
            run_from_wandb, 
            model_kwarg_in_trainer=model_kwarg_in_trainer,
            model_sub_config_name_base=model_sub_config_name_base,
            trainer_default_module_key=trainer_default_module_key,
            add_model_module_to_architecture_default_module=add_model_module_to_architecture_default_module,
            trainer_activation_method=trainer_activation_method,
            provide_prng_key_to_trainer_activation_method=provide_prng_key_to_trainer_activation_method,
            additional_trainer_default_modules = [project_utils, optax],
            **kwargs
        ),
        count=count
    )
    wandb.teardown()
    print(f"Finished {count} runs for {full_sweep_id}")

if __name__ == '__main__':
    try:
        fire.Fire(main)
    except Exception as e:
        print(f"Got the following exception: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
