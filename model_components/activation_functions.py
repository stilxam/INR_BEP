""" 
Module with activation functions for inr layers.
"""
from typing import Union
import jax
from jax import numpy as jnp

def unscaled_gaussian_bump(*x:jax.Array, inverse_scale:Union[float, jax.Array]):
    """ 
    e^(sum_{x' in x}-|inverse_scale*x'|^2)

    :param x: sequence of arrays for which to calculate the gaussian bump
    :returns: the product of the gaussian bumps (computed as a sum in log-space)
    """
    x = jnp.stack(x, axis=0)
    if jnp.isrealobj(x):
        scaled_x = inverse_scale*x
    else:
        scaled_x = jnp.abs(inverse_scale*x)
    return jnp.exp(-jnp.sum(jnp.square(scaled_x), axis=0))
