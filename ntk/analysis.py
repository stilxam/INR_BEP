from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import jit



def get_NTK_ntvp(apply_fn: Callable) -> Callable:
    """Get NTK computation function."""
    kwargs = dict(f=apply_fn, trace_axes=(), vmap_axes=0)
    return jit(
        nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS
        )
    )

def decompose_ntk(ntk: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Decompose NTK into eigenvalues and eigenvectors."""
    eigvals, eigvecs = jnp.linalg.eigh(ntk)
    rescaled_eigvals = jnp.flipud(eigvals) / jnp.min(jnp.abs(eigvals))
    return jnp.flipud(eigvals), jnp.flipud(eigvecs.T), rescaled_eigvals


def analyze_fft(ntk: jnp.ndarray) -> jnp.ndarray:
    """Compute FFT of NTK."""
    fft_ntk = jnp.fft.fft2(ntk)
    return jnp.log(jnp.abs(jnp.fft.fftshift(fft_ntk)) + 2e-5)

def analyze_fft_spectrum(magnitude_spectrum: jnp.ndarray) -> Dict[str, float]:
    """Analyze FFT magnitude spectrum of the NTK."""
    mean_magnitude = jnp.mean(magnitude_spectrum)
    std_magnitude = jnp.std(magnitude_spectrum)
    diagonal_strength = jnp.mean(jnp.diag(magnitude_spectrum))
    off_diagonal = magnitude_spectrum - jnp.diag(jnp.diag(magnitude_spectrum))
    off_diagonal_mean = jnp.mean(off_diagonal)

    n = magnitude_spectrum.shape[0]
    quarter = n // 4
    corners = (
        magnitude_spectrum[:quarter, :quarter].mean()
        + magnitude_spectrum[-quarter:, -quarter:].mean()
    )

    return {
        "mean_magnitude": float(mean_magnitude),
        "std_magnitude": float(std_magnitude),
        "diagonal_strength": float(diagonal_strength),
        "off_diagonal_mean": float(off_diagonal_mean),
        "high_freq_content": float(corners),
    }
