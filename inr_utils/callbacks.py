""" 
Module containing callbacks for the training.train_inr function.
"""
import abc
import pprint
from typing import Callable, Union, Any

import jax
from jax import numpy as jnp

from common_dl_utils.metrics import MetricCollector
from common_dl_utils.type_registry import register_type


@register_type
class Callback(abc.ABC):
    @abc.abstractmethod
    def __call__(self, step, loss, inr, optimizer_state)->Any:
        pass


def print_loss(step, loss, inr, optimizer_state, after_every=200):
    if (step+1)%after_every == 0:
        print(f"Loss at step {step+1} is {loss}.")

# ===========================================================================================
# the callbacks below return dictionaries containing logs that can be collected by other callbacks

class MetricCollectingCallback(Callback):
    def __init__(self, metric_collector:MetricCollector):
        self.metric_collector = metric_collector
    def __call__(self, step, loss, inr, optimizer_state):
        return self.metric_collector.on_batch_end(inr=inr, loss=loss, optimizer_state=optimizer_state, step=step)

def report_loss(step, loss, inr, optimizer_state):
    return {'loss': loss}


# ===========================================================================================
# the callback below raises an error if the loss contains NaNs
# this way we don't keep training while only passing around NaNs
def raise_error_on_nan(step, loss, inr, optimizer_state):
    if jnp.any(jnp.isnan(loss)):
        raise RuntimeError(f"NaN occurred in loss at step {step}.")

# ===========================================================================================
# the one callback to rule them all

class ComposedCallback(Callback):
    def __init__(
            self, 
            *callbacks:Union[Callback, Callable], 
            use_wandb:bool, 
            show_logs:bool, 
            display_func:Callable=pprint.pprint
            ):
        self._wandb = None
        if use_wandb: # wanted to keep wandb an optional dependency
            import wandb
            self._wandb = wandb
        self.callbacks = list(callbacks)
        self.show_logs = show_logs
        self.display_func = display_func
    
    def __call__(self, step, loss, inr, optimizer_state):
        logs = {}
        for callback in self.callbacks:
            log = callback(step, loss, inr, optimizer_state)
            if log is not None:
                logs.update(log)
        if self._wandb is not None and logs:
            self._wandb.log(logs)
        if self.show_logs and logs:
            self.display_func(logs)
