""" 
Module with activation functions for inr layers.
"""
from typing import Union
import jax
from jax import numpy as jnp


def unscaled_gaussian_bump(*x: jax.Array, inverse_scale: Union[float, jax.Array]):
    """ 
    e^(sum_{x' in x}-|inverse_scale*x'|^2)

    :param x: sequence of arrays for which to calculate the gaussian bump
    :returns: the product of the gaussian bumps (computed as a sum in log-space)
    """
    x = jnp.stack(x, axis=0)
    if jnp.isrealobj(x):
        scaled_x = inverse_scale * x
    else:
        scaled_x = jnp.abs(inverse_scale * x)
    return jnp.exp(-jnp.sum(jnp.square(scaled_x), axis=0))


def real_gabor_wavelet(x: jax.Array, s0: Union[float, jax.Array], w0: Union[float, jax.Array]):
    """ 
    The WIRE paper (https://arxiv.org/pdf/2301.05187) states that the R->R version of the gabor wavelet is
    \sigma(x) = sin(w_0*x[0])*exp(-(s_0 *x[1])^2).
    However, their code (https://github.com/vishwa91/wire/blob/main/modules/wire.py),
    implements it as, \sigma(x) = cos(w_0 *x[0]) * exp(-(s_0 *x[1])^2).
    Hence, we follow the code.

    :parameter x: two `jax.Array`s to be fed to this activation function, x = [frequency, scale]

    :parameter s0: inverse scale used in the radial part of the wavelet (s_0 in the paper)
        keyword only
    :parameter w0: w0 parameter used in the rotational art of the wavelet (\omega_0 in the paper)
        keyword only
    :return: a `jax.Array` with the same shape as x[0] or x[1]
    """
    omega = w0 * x[0]
    scale = s0 * x[1]
    return jnp.cos(omega) * jnp.exp(-jnp.square(scale))


def complex_gabor_wavelet(x: jax.Array, s0: Union[float, jax.Array], w0: Union[float, jax.Array]):
    """
    Implements the WIRE activation function
    that is \sigma(x) = exp(j w_0 x)* exp(-|s_0 x|^2)
    from https://arxiv.org/pdf/2301.05187

    :parameter x: a bunch of `jax.Array`s to be fed to this activation function
        var positional
    :parameter s0: inverse scale used in the radial part of the wavelet (s_0 in the paper)
        keyword only
    :parameter w0: w0 parameter used in the rotational art of the wavelet (\omega_0 in the paper)
        keyword only
    :return: a `jax.Array` with a shape determined by broadcasting all elements of x to tha same shape
    """
    omega = w0 * x
    scale = s0 * x
    return jnp.exp(-jnp.square(jnp.abs(scale)) + 1j * omega)
