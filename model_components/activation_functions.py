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


def real_wire(*x: jax.Array, s0:Union[float, jax.Array], w0:Union[float, jax.Array]):
    """ 
    Implements a real version of WIRE-nD
    that is sin(w0*x[0])*exp(-\sum_{x' in x}|inverse_scale*x'|^2)
    from https://arxiv.org/pdf/2301.05187

    :parameter x: a bunch of `jax.Array`s to be fed to this activation function
        var positional
    :parameter s0: inverse scale used in the radial part of the wavelet (s_0 in the paper)
        keyword only
    :parameter w0: w0 parameter used in the rotational art of the wavelet (\omega_0 in the paper)
        keyword only
    :return: a `jax.Array` with a shape determined by broadcasting all elements of x to tha same shape
    """
    radial_part = unscaled_gaussian_bump(*x, inverse_scale=s0)
    rotational_part = jnp.sin(w0*x[0])
    return rotational_part*radial_part



def WIRE(x: jax.Array, s0:Union[float, jax.Array], w0:Union[float, jax.Array]):
    """
    Implements the WIRE activation function
    that is sin(w0*x[0])*exp(-\sum_{x' in x}|inverse_scale*x'|^2)
    from https://arxiv.org/pdf/2301.05187

    :parameter x: a bunch of `jax.Array`s to be fed to this activation function
        var positional
    :parameter s0: inverse scale used in the radial part of the wavelet (s_0 in the paper)
        keyword only
    :parameter w0: w0 parameter used in the rotational art of the wavelet (\omega_0 in the paper)
        keyword only
    :return: a `jax.Array` with a shape determined by broadcasting all elements of x to tha same shape
    """
    omega = w0*x
    scale = s0*x
    return jnp.exp(-jnp.square(jnp.abs(scale)) + 1j*omega)

