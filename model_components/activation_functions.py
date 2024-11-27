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


def multidimensional_complex_gabor_wavelet(x: jax.Array, s0: Union[float, jax.Array], w0: Union[float, jax.Array]):
    """
    Implements the WIRE activation function as per the paper
    y_m = \psi(W^1_m y_{m-1} + b^1_m) exp(-\sum_{i=2}^{m} |s_0 (W^i_m y_{m-1} + b^i_m)|^2)

    This represents a custom activation function where:
    - ψ is a wavelet function with parameters w0 and s0.
    - W and b are lists of weight matrices and bias vectors respectively.
    - y_{m-1} is the input from the previous layer.
    - The first term is the wavelet function applied to the linear transformation of the input.
    - The second term is an exponential decay based on the sum of squared scaled linear transformations of the input.

    in this case, the matrix multiplication has already been done, so we only need to apply the non-linearity.
    this yields the following activation function:
    y_m = \psi(h[1]) exp(-\sum_{i=2}^{m} |s_0 (h[i])|^2)
    """
    gab = complex_gabor_wavelet(x[0], s0=s0, w0=w0)
    for i in range(1, len(x)):
        content = jnp.square(jnp.abs(s0 * x[i]))
        gab = gab * jnp.exp(-content)
    return gab


def two_d_complex_gabor_wavelet(x: jax.Array, s0: Union[float, jax.Array], w0: Union[float, jax.Array]):
    """
    Implements the 2D WIRE activation function as per the github code
    that is \sigma(x) = exp(j w_0 x[0])* exp(-s_0^2 * (|x[0|^2 + |x[1]|^2))
    from https://github.com/vishwa91/wire/blob/main/modules/wire2d.py
    :parameter x: 2 `jax.Array`s to be fed to this activation function
    :parameter s0: inverse scale used in the radial part of the wavelet (s_0 in the paper)
    :parameter w0: w0 parameter used in the rotational art of the wavelet (\omega_0 in the paper)
    """
    freq = jnp.exp(1j*w0*x[0])
    arg = jnp.square(jnp.abs(x[0])) + jnp.square(jnp.abs(x[1]))
    gaus = jnp.exp(-jnp.square(s0)*arg)
    return freq* gaus

