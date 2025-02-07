import wandb

from ntk.sweep import main_sweep
from ntk.config import get_sweep_configuration

if __name__ == "__main__":
    sweep_config = get_sweep_configuration()
    sweep_id = wandb.sweep(sweep_config, project="ntk-analysis")
    wandb.agent(sweep_id, main_sweep)
