""" 
Module with activation functions for inr layers.
"""
from typing import Union, Tuple
import jax
from jax import numpy as jnp


def unscaled_gaussian_bump(*x: jax.Array, inverse_scale: Union[float, jax.Array]):
    """ 
    e^(sum_{x' in x}-|inverse_scale*x'|^2)

    :param x: sequence of arrays for which to calculate the gaussian bump
    :returns: the product of the gaussian bumps (computed as a mean in log-space)
    """
    x = jnp.stack(x, axis=0)
    if jnp.isrealobj(x):
        scaled_x = inverse_scale * x
    else:
        scaled_x = jnp.abs(inverse_scale * x)
    return jnp.exp(-jnp.mean(jnp.square(scaled_x), axis=0))  # mean instead of sum so that we don't have to change the initialization scheme


def real_gabor_wavelet(x: tuple[jax.Array, jax.Array], s0: Union[float, jax.Array], w0: Union[float, jax.Array]):
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


# def complex_gabor_wavelet(x: jax.Array, s0: Union[float, jax.Array], w0: Union[float, jax.Array]):
#     """
#     Implements the WIRE activation function
#     that is \sigma(x) = exp(j w_0 x)* exp(-|s_0 x|^2)
#     from https://arxiv.org/pdf/2301.05187

#     :parameter x: a bunch of `jax.Array`s to be fed to this activation function
#         var positional
#     :parameter s0: inverse scale used in the radial part of the wavelet (s_0 in the paper)
#         keyword only
#     :parameter w0: w0 parameter used in the rotational art of the wavelet (\omega_0 in the paper)
#         keyword only
#     :return: a `jax.Array` with a shape determined by broadcasting all elements of x to tha same shape
#     """
#     omega = w0 * x
#     scale = s0 * x
#     return jnp.exp(-jnp.square(jnp.abs(scale)) + 1j * omega)

def complex_gabor_wavelet(*x: jax.Array, s0:Union[float, jax.Array], w0: Union[float, jax.Array])->jax.Array:
    """ 
    Implements the n-dimensional WIRE activation function as per the 2D extension.
    :parameter x: one jax.Array per dimension
        should all have the same size
    :parameter s0: inverse scale parameter of the wavelet (s_0 in the paper)
    :parameter w0: frequency scale parameter of the wavelet (\omega_0 in the paper)
    :returns: the computed wavelet as a JAX array
    """
    # Frequency modulation ( w0 applies only to the first coordinate)
    freq = jnp.exp(1j * w0 * x[0])
    # Radial part: Gaussian envelope
    x = jnp.stack(x, axis=0)
    squared_abs = jnp.mean(jnp.square(jnp.abs(x)), axis=0)  # mean instead of sum so that we don't have to change the initialization scheme
    gaussian_envelope = jnp.exp(-jnp.square(s0)*squared_abs)
    return freq * gaussian_envelope


# def two_d_complex_gabor_wavelet(*x: jax.Array, s0: Union[float, jax.Array], w0: Union[float, jax.Array]):  # same results as complex_gabor_wavelet for 2d
#     """
#     Implements the 2D WIRE activation function as per the github code
#     that is \sigma(x) = exp(j w_0 x[0])* exp(-s_0^2 * (|x[0|^2 + |x[1]|^2))
#     from https://github.com/vishwa91/wire/blob/main/modules/wire2d.py
#     :parameter x: 2 `jax.Array`s to be fed to this activation function
#     :parameter s0: inverse scale used in the radial part of the wavelet (s_0 in the paper)
#     :parameter w0: w0 parameter used in the rotational art of the wavelet (\omega_0 in the paper)
#     """
#     freq = jnp.exp(1j*w0*x[0])
#     arg = jnp.square(jnp.abs(x[0])) + jnp.square(jnp.abs(x[1]))
#     gaus = jnp.exp(-jnp.square(s0)*arg)
#     return freq * gaus

def finer_activation(x, w0):
    """
    FINER activation function: sin((|x| + 1) * x)
    :param x: input array for activation
    :param w0: frequency scaling factor
    :return: output array after applying variable-periodic function
    """
    return jnp.sin((jnp.abs(x) + 1) * w0 * x)
