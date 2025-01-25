""" 
Some common loss functions for training INRs.
"""
from typing import Optional, Callable, Union
import functools

import jax
from jax import numpy as jnp
from jaxtyping import PyTree  # really just Any
import equinox as eqx

from inr_utils import nerf_utils
from model_components.inr_modules import NeRF

LossEvaluator = Union[Callable[[eqx.Module, PyTree, Optional[eqx.nn.State]], jax.Array], eqx.Module]
LossEvaluator.__doc__ = """
A LossEvaluator is a function that should take
* the model (eqx.Module) that is to be evaluated
* a PyTree containing (possibly among other things) the input values for the model
it should have *scalar* output.
"""


def mse_loss(pred_val: jax.Array, true_val: jax.Array):
    return jnp.mean(jnp.sum(jnp.square(pred_val - true_val), axis=-1))


def scaled_mse_loss(pred_val: jax.Array, true_val: jax.Array, eps=1e-6):
    """ 
    A scaled version of the mse loss, where the loss on each batch gets scaled 
    so that 1 means you correctly predict the mean of the true values over that batch

    :parameter pred_val: the predicted values
    :parameter true_val: the true values
    :parameter eps: a small value to avoid division by zero
    """
    mse = mse_loss(pred_val, true_val)
    mean_true_val = true_val.mean(axis=0, keepdims=True)
    scaling = mse_loss(mean_true_val, true_val)
    return mse / (scaling + eps)


def psnr_loss(pred_val: jax.Array, true_val: jax.Array, max_val=1.):
    """
    Peak Signal-to-Noise Ratio loss
    :param pred_val: the predicted values
    :param true_val: the true values
    :param max_val: the maximum value of the input data
    """
    mse = mse_loss(pred_val, true_val)
    return 20 * jnp.log10(max_val) - 10 * jnp.log10(mse)



# ==========================================================================
# loss evaluators

class PointWiseLossEvaluator(eqx.Module):
    """
    Evaluates an inr and a target function on a batch of input coordinates
    and computes the loss between the results of those two.

    :param target_function: function to be approximated by the INR
    :param loss_function: function to compute the loss based on pred_val and true_val
        should have the following signature
        :param pred_val: jax.Array of predicted values
        :param true_val: jax.Array of target values with same shape as pred_val
        :return: scalar jax.Array containing the loss value
    :param state_update_function: optional callable to update the (optional) state
        E.G. useful for implementing the progressive training in the integer lattice mapping
        NB the state is only updated by this, not by the INR itself
    """
    target_function: Union[Callable, eqx.Module]
    loss_function: Callable
    state_update_function: Optional[Callable] = None

    def __call__(self, inr: eqx.Module, locations: jax.Array, state: Optional[eqx.nn.State] = None):
        if state is not None:
            pred_val, _ = jax.vmap(inr, (0, None))(locations, state)
        else:
            pred_val = jax.vmap(inr)(locations)
        true_val = jax.vmap(self.target_function)(locations)

        if self.state_update_function is not None:  # update state if necessary
            # this is e.g. for the progressive training in the integer lattice mapping
            state = self.state_update_function(state, inr)

        return self.loss_function(pred_val, true_val), state


class PointWiseGradLossEvaluator(eqx.Module):
    target_function: Union[Callable, eqx.Module]
    loss_function: Callable
    state_update_function: Optional[Callable]

    def __init__(
            self,
            target_function: Callable,
            loss_function: Callable,
            take_grad_of_target_function: bool,
            state_update_function: Optional[Callable] = None
    ):
        """ 
        evaluates *the gradient of* an inr and a target function on a batch of input coordinates
        and computes the loss between the results of those two.

        :param target_function: function to be approximated by the INR
        :param loss_function: function to compute the loss based on pred_val and true_val
            should have the following signature
            :param pred_val: jax.Array of predicted values
            :param true_val: jax.Array of target values with same shape as pred_val
            :return: scalar jax.Array containing the loss value
        :param take_grad_of_target_function: 
            if True, we use automatic differentiation to get the gradient of the target function
            if False, we assume that the output of the target function is itself the relevant gradient and no automatic differentiation is applied.
        :param state_update_function: optional callable to update the (optional) state
            E.G. useful for implementing the progressive training in the integer lattice mapping
            NB the state is only updated by this, not by the INR itself
        """
        self.loss_function = loss_function
        if take_grad_of_target_function:
            self.target_function = eqx.filter_grad(target_function)
        else:
            self.target_function = target_function
        self.state_update_function = state_update_function

    def __call__(self, inr: eqx.Module, locations: jax.Array, state: Optional[eqx.nn.State] = None):
        if state is not None:
            inr_grad = eqx.filter_grad(inr, has_aux=True)
            pred_val, _ = jax.vmap(inr_grad, (0, None))(locations, state)
        else:
            inr_grad = eqx.filter_grad(inr)
            pred_val = jax.vmap(inr_grad)(locations)
        true_val = jax.vmap(self.target_function)(locations)

        if self.state_update_function is not None:  # update state if necessary
            # this is e.g. for the progressive training in the integer lattice mapping
            state = self.state_update_function(state, inr)

        return self.loss_function(pred_val, true_val), state


class NeRFLossEvaluator(nerf_utils.Renderer):

    parallel_batch_size: int
    state_update_function: Optional[Callable] = None

    def __call__(self, nerf_model: NeRF, batch:tuple[jax.Array, jax.Array, jax.Array, jax.Array], state:Optional[eqx.nn.State] = None):
        ray_origins, ray_directions, prng_key_or_keys, ground_truth = batch
        # give each batch element its own key
        if prng_key_or_keys.shape[0] != ray_origins.shape[0]:
            prng_keys = jax.random.split(prng_key_or_keys, num=ray_origins.shape[0])
        else:
            prng_keys = prng_key_or_keys
        
        # render each pixel
        if state is not None:
            results = jax.lax.map(
                lambda ro_rd_k: self.render_nerf_pixel(nerf_model, *ro_rd_k, state),
                (ray_origins, ray_directions, prng_keys),
                batch_size=self.parallel_batch_size
            )
        else:
            results = jax.lax.map(
                lambda ro_rd_k: self.render_nerf_pixel(nerf_model, *ro_rd_k),
                (ray_origins, ray_directions, prng_keys),
                batch_size=self.parallel_batch_size
            )
        
        coarse_predictions = results["coarse_rgb"]
        fine_predictions = results["fine_rgb"]

        if self.state_update_function is not None:  # update state if necessary
            # this is e.g. for the progressive training in the integer lattice mapping
            state = self.state_update_function(state, nerf_model)

        return jnp.square(coarse_predictions - ground_truth).mean() + jnp.square(fine_predictions - ground_truth).mean(), state
