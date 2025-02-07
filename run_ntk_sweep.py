import wandb

from ntk.sweep import main_sweep, setup_sweep_config
from ntk.config import get_sweep_configuration, get_2d_sweep_configuration

if __name__ == "__main__":
    # sweep_config = get_sweep_configuration()
    sweep_config = get_2d_sweep_configuration()
    sweep_id = wandb.sweep(sweep_config, project="ntk-analysis")
    wandb.agent(sweep_id, main_sweep)
