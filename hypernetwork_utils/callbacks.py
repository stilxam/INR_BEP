""" 
Callbacks for training. The trainer in hypernetwork_utils.training allows for an after step callback and for an after epoch callback.
To use multiple callbacks, you can put multiple callbacks together using ComposedCallback.
"""
from typing import Callable, Union, Optional
import abc

import jax
from jax import numpy as jnp
import equinox as eqx
import optax

from common_jax_utils.types import register_type

@register_type
class Callback(abc.ABC):
    """ 
    Abstract baseclass for callbacks specifying their call signature.
    Note that the Trainer from hypernetwork_utils.training also allows for functions as callbacks.
    Those functions however should have the same (or a compatible) call signature.
    """
    @abc.abstractmethod
    def __call__(
            self, 
            *, 
            step_within_epoch:Optional[int], 
            epoch:int, loss:Union[float, jax.Array], 
            hypernetwork:eqx.Module, 
            optimizer_state:optax.OptState, 
            report:Optional[dict]
        )->None:
        """ 
        :parameter step_within_epoch: for after step callbacks only: what step in the epoch was just finished
        :parameter epoch: current epoch
        :parameter hypernetwork: the hypernetwork being trained
        :parameter optimizer_state: the state of the (optax) optimizer
        :parameter report: dictionary containing values of metrics, loss, etc.
        """
        pass

class ComposedCallback(Callback):
    """ 
    Container class to group multiple callbacks into a single callback.
    """
    def __init__(
            self, 
            *callbacks: Union[Callback, Callable]
            ):
        self.callbacks = callbacks

    def __call__(self, *, step_within_epoch:Optional[int], epoch:int, loss:Union[float, jax.Array], hypernetwork:eqx.Module, optimizer_state:optax.OptState, report:Optional[dict]):
        for callback in self.callbacks:
            callback(
                step_within_epoch=step_within_epoch,
                epoch=epoch,
                loss=loss,
                hypernetwork=hypernetwork,
                optimizer_state=optimizer_state,
                report=report
            )

def raise_error_on_nan(step_within_epoch, epoch, loss, **kwargs):
    """ 
    Callback that can be useful for stopping training whenever a NaN has occurred 
    """
    if jnp.any(jnp.isnan(loss)):
        raise RuntimeError(f"NaN occurred in loss at step {step_within_epoch} of epoch {epoch}.")

def print_loss(step_within_epoch, loss, *, after_every=200, **kwargs):
    """ 
    Callback for reporting loss to stdout
    """
    if step_within_epoch % after_every == 0:
        print(f"    Loss at step {step_within_epoch} is {loss}.")
