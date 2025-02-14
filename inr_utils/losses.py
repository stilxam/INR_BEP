""" 
Some common loss functions for training INRs.
"""
from typing import Optional, Callable, Union

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

    return mse/(scaling + eps)


def cosine_similarity(a:jax.Array, b:jax.Array)->jax.Array:
    """
    Compute the cosine similarity between arrays a and b
    """
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def sdf_loss(gt_normals, y, y_pred, y_grad) -> jax.Array:
    """
    Compute the loss for the sdf values, the intersection values, the normal values, and the gradient values
    as per the SIREN paper and code (https://github.com/vsitzmann/siren/blob/master/loss_functions.py)

    if the model uses relu, y_grad = 0 can break the loss function and make gradients loss nan
    :param gt_normals: ground truth normals
    :param y: ground truth sdf values
    :param y_pred: predicted sdf values
    :param y_grad: gradients of the sdf values
    :return: loss_array, containing the loss values for sdf, inter, normal, and grad
    """

    sdf_constraint = jnp.where(y != -1, y_pred, jnp.zeros_like(y_pred))
    inter_constraint = jnp.where(y != -1, jnp.zeros_like(y_pred), jnp.exp(-1e2 * jnp.abs(y_pred)))
    cosine_sim = jax.vmap(cosine_similarity)(y_grad, gt_normals)
    if len(cosine_sim.shape) == 1:
        y = jnp.expand_dims(y, axis=1)

    normal_constraint = jnp.where(y != -1, 1 - cosine_sim, jnp.zeros_like(y_grad[:, 0]))
    gradient_constraint = jnp.abs(jnp.linalg.norm(y_grad) - 1)

    loss_array = jnp.array([
        #jnp.abs(jnp.mean(sdf_constraint)) * 3000,  # I left it here so you can remove this bane of your existence for the past weeks yourself @stilxam TODO
        jnp.mean(jnp.abs(sdf_constraint)) * 3000,
        jnp.mean(inter_constraint) * 3000,
        jnp.mean(normal_constraint) * 100,
        jnp.mean(gradient_constraint) * 50
    ])
    return jnp.sum(loss_array)



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
            target_function: Union[Callable, eqx.Module],
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
            self.target_function = eqx.filter_grad(lambda x: target_function(x).squeeze())
        else:
            self.target_function = target_function
        self.state_update_function = state_update_function

    @staticmethod
    def wrap_inr(inr):
        def wrapped(*args, **kwargs):
            out = inr(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            return out
        return wrapped

    def __call__(self, inr: eqx.Module, locations: jax.Array, state: Optional[eqx.nn.State] = None):
        inr = self.wrap_inr(inr)
        if state is not None:
            inr_grad = eqx.filter_grad(inr)
            pred_val = jax.vmap(inr_grad, (0, None))(locations, state)
        else:
            inr_grad = eqx.filter_grad(inr)
            pred_val = jax.vmap(inr_grad)(locations)

        true_val = jax.vmap(self.target_function)(locations)

        if self.state_update_function is not None:  # update state if necessary
            # this is e.g. for the progressive training in the integer lattice mapping
            state = self.state_update_function(state, inr)

        return self.loss_function(pred_val, true_val), state
    

class SDFLossEvaluator(eqx.Module):
    """ 
    Evaluates the loss for the sdf values, the intersection values, the normal values, and the gradient values
    as per the SIREN paper and code (https://github.com/vsitzmann/siren/blob/master/loss_functions.py)
    """
    state_update_function: Optional[Callable] = None

    def __call__(self, inr: eqx.Module, batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array], state: Optional[eqx.nn.State] = None):
        """
        Compute the SDF loss for a batch of points.

        Args:
            inr: The INR module representing the SDF
            batch: Tuple of (coords, normals, sdf_values) from SDFSampler
            state: Optional state for stateful INRs

        Returns:
            Tuple of (loss, state) where loss is a jax.Array containing the individual loss components
            [sdf_loss, intersection_loss, normal_loss, gradient_loss]
        """
        coords, gt_normals, gt_sdf_values = batch
        
        inr_wrapped = self.wrap_inr(inr)

        if state is not None:
            
            inr_grad = eqx.filter_grad(inr_wrapped)
            grad_val = jax.vmap(inr_grad, (0, None))(coords, state)
            pred_val = jax.vmap(inr_wrapped, (0, None))(coords, state)
        else:
            inr_grad = eqx.filter_grad(inr_wrapped)
            grad_val = jax.vmap(inr_grad)(coords)
            pred_val = jax.vmap(inr_wrapped)(coords)

        loss = sdf_loss(gt_normals, gt_sdf_values, pred_val, grad_val)
        if self.state_update_function is not None:  # update state if necessary
            # this is e.g. for the progressive training in the integer lattice mapping
            state = self.state_update_function(state, inr)
            
        return loss, state

    @staticmethod
    def wrap_inr(inr):
        def wrapped(*args, **kwargs):
            out = inr(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            return out

        return wrapped

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
        if len(time_points.shape) < 3:
            time_points = time_points[..., None]
        

        # Evaluate INR at time points - vmap over batch and window dimensions
        #inr_values = jax.vmap(lambda t: jax.vmap(lambda ti: inr(ti))(t))(time_points)
        if state is None:
            inr_values = jax.vmap(jax.vmap(inr))(time_points)  #should also work, but might need to adjust in and out axes
        else:
            inr_values, _ = jax.vmap(jax.vmap(inr, (0, None)), (0, None))(time_points, state)
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

        if self.state_update_function is not None:  # update state if necessary
            # this is e.g. for the progressive training in the integer lattice mapping
            state = self.state_update_function(state, inr)

        return total_loss, state


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


