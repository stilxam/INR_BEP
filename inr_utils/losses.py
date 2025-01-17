""" 
Some common loss functions for training INRs.
"""
from typing import Optional, Callable, Union

import jax
from jax import numpy as jnp
from jaxtyping import PyTree  # really just Any

import equinox as eqx


LossEvaluator = Union[Callable[[eqx.Module, PyTree, Optional[eqx.nn.State]], jax.Array], eqx.Module]
LossEvaluator.__doc__ = """
A LossEvaluator is a function that should take
* the model (eqx.Module) that is to be evaluated
* a PyTree containing (possibly among other things) the input values for the model
it should have *scalar* output.
"""

def mse_loss(pred_val: jax.Array, true_val: jax.Array):
    return jnp.mean(jnp.sum(jnp.square(pred_val - true_val), axis=-1))

def scaled_mse_loss(pred_val:jax.Array, true_val:jax.Array, eps=1e-6):  
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
    return mse/(scaling + eps)


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

    def __call__(self, inr: eqx.Module, locations: jax.Array, state:Optional[eqx.nn.State]=None):
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
            target_function:Callable, 
            loss_function:Callable, 
            take_grad_of_target_function:bool, 
            state_update_function:Optional[Callable]=None
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

    def __call__(self, inr:eqx.Module, locations: jax.Array, state:Optional[eqx.nn.State]=None):
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

class SoundLossEvaluator(eqx.Module):
    """ 
    Evaluate the loss on a sound by combining time domain and frequency domain losses.
    The time domain loss measures how well the INR matches the original pressure values.
    The frequency domain loss measures how well the INR matches the frequency components
    obtained via FFT.

    :parameter time_domain_weight: Weight for the loss in time domain
    :parameter frequency_domain_weight: Weight for the loss in frequency domain (between FFT arrays)
    :parameter state_update_function: Optional function to update the state of the loss evaluator
    """
    time_domain_weight: float
    frequency_domain_weight: float 
    state_update_function: Optional[Callable] = None

    def __call__(self, inr: eqx.Module, locations: tuple[jax.Array, jax.Array], state: Optional[eqx.nn.State] = None) -> tuple[float, Optional[eqx.nn.State]]:
        """
        Evaluate combined time and frequency domain loss between INR output and true signal.

        :param inr: The INR module to evaluate
        :param locations: Tuple of (time_points, pressure_values) from SoundSampler
        :param state: Optional state for stateful INRs
        :return: Tuple of (total_loss, updated_state) if state provided, else (total_loss, None)
        """
        time_points, pressure_values = locations

        # Update state if needed
        if state is not None and self.state_update_function is not None:
            inr, state = self.state_update_function(inr, state)

        # Evaluate INR at time points - vmap over batch and window dimensions
        #inr_values = jax.vmap(lambda t: jax.vmap(lambda ti: inr(ti))(t))(time_points)
        inr_values = jax.vmap(jax.vmap(inr)(time_points))  #should also work, but might need to adjust in and out axes
        inr_values = jnp.squeeze(inr_values, axis=-1)  # Remove last dimension of shape 1

        # Calculate time domain MSE loss
        time_loss = mse_loss(inr_values, pressure_values)

        # Calculate FFT for both signals - vmap over batch dimension
        inr_fft = jax.vmap(jnp.fft.fft)(inr_values)
        true_fft = jax.vmap(jnp.fft.fft)(pressure_values)

        # Calculate frequency domain loss using magnitudes
        freq_loss = mse_loss(jnp.abs(inr_fft), jnp.abs(true_fft))

        # Combine losses with weights
        total_loss = self.time_domain_weight * time_loss + self.frequency_domain_weight * freq_loss

        return total_loss, state
        

    


