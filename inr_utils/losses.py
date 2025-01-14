""" 
Some common loss functions for training INRs.
"""
import jax
from jax import numpy as jnp

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
        inr_values = jax.vmap(lambda t: jax.vmap(lambda ti: inr(ti))(t))(time_points)
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

        return total_loss, state if state is not None else None
        

    


