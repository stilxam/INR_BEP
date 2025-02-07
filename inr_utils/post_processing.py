from typing import Optional
import pprint
import os

import jax
import equinox as eqx
import wandb
import numpy as np
from matplotlib import pyplot as plt

import common_dl_utils as cdu
import common_jax_utils as cju

from inr_utils import metrics


class PostProcessor(eqx.Module):
    metrics: list[metrics.Metric]
    storage_directory: str
    wandb_kwargs: dict

    def __call__(self, 
                 results,  
                 experiment_parameters, 
                 experiment_config, 
                 key:Optional[jax.Array]=None, 
                 config_file_path:Optional[str]=None,
                 ):
        with wandb.init(**self.wandb_kwargs) as run:
            print(f"Postprocessing results for run {run.name}  with parameters:\n{pprint.pformat(experiment_parameters)}\n")
            run.log(cdu.config_creation.make_flat_config(experiment_parameters))
            run.log({'config': pprint.pformat(experiment_config)})
            if config_file_path is not None:
                run.log_artifact(config_file_path, name="config_file")
            
            inr, optimizer_state, state, losses = results
            losses = np.asarray(losses)
            log_dict = dict(
                final_loss = losses[-1],
                losses = losses,
            )

            key_gen = cju.key_generator(key)
            
            # compute metrics
            print("Computing metrics")
            for metric in self.metrics:
                log_dict.update(
                    metric.compute(inr=inr, optimizer_state=optimizer_state, state=state, key=next(key_gen), **experiment_parameters)
                )
            
            # plot loss
            print("Plotting loss")
            fig, ax = plt.subplots()
            if np.all(losses>0):
                ax.set_yscale('log')
            ax.plot(losses)
            log_dict["loss_plot"] = fig
            run.log(log_dict)

            # store the model
            print("Storing model")
            if not os.path.exists(self.storage_directory):
                os.makedirs(self.storage_directory, exist_ok=True)
            storage_path = f"{self.storage_directory}/{run.name}.eqx"
            eqx.tree_serialise_leaves(storage_path, inr)
            wandb.log_model(path=storage_path, name=f"{run.name}.eqx")
            print(f"    Stored model at {storage_path}")
            print(f"Finished postprocessing {run.name}.")
