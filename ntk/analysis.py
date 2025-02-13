import scipy
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import jit



def get_NTK_ntvp(apply_fn: Callable) -> Callable:
    """Get NTK computation function."""

    # apply_fn = jax.grad(apply_fn)
    kwargs = dict(f=apply_fn, trace_axes=(), vmap_axes=0)
    return jit(
        nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS
            # **kwargs, implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION
            # **kwargs, implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES
        )
    )

def decompose_ntk(ntk: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Decompose NTK into eigenvalues and eigenvectors."""
    # eigvals, eigvecs = jnp.linalg.eigh(ntk)

    eigvals, eigvecs = scipy.sparse.linalg.eigsh(jax.device_get(ntk), k=ntk.shape[0]-1)

    rescaled_eigvals = jnp.flipud(eigvals) / jnp.min(jnp.abs(eigvals))
    # condition_number = jnp.linalg.cond(ntk)
    condition_number = eigvals.max() / jnp.abs(eigvals.min())
    # assert jnp.all(jnp.isfinite(eigvals))
    # assert not jnp.any(jnp.isnan(eigvecs))
    # if jnp.any(jnp.isnan(eigvecs)):
    #     eigvecs = jnp.zeros_like(eigvecs)
    # if not jnp.all(jnp.isfinite(eigvals)):
    #     eigvals = jnp.zeros_like(eigvals)
    #     condition_number = -1


    return jnp.flipud(eigvals), jnp.flipud(eigvecs.T), rescaled_eigvals, condition_number


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




def make_map(k):
    idx = jnp.arange(k)+1
    return jnp.abs(idx - idx[:, None])


def measure_of_diagonal_strength(ntk: jax.Array, map_kwarg:int =0, exp_scale:float =1.0):
    '''
    Measures how much of a matrix's "energy" is concentrated on its diagonal.
    
    This function computes a ratio between the weighted sum of squared off-diagonal elements
    and the mean squared value of diagonal elements. The weighting scheme is controlled by map_kwarg:
    
    Args:
        ntk: An nxn symmetric matrix to analyze
        map_kwarg: Integer controlling the weighting scheme:

    Returns:
        float: A measure of diagonal strength. Lower values indicate more diagonal-dominant matrices.
              A perfectly diagonal matrix would return 0.
    '''
    size = ntk.shape[0]
    if map_kwarg == 0:
        weights = make_map(size)+1
    elif map_kwarg == 1:
        weights = jnp.ones((size, size))
    elif map_kwarg == 2:
        weights = 1/(make_map(size)+1)
    elif map_kwarg == -1:
        weights = (make_map(size)+1)*jnp.exp(-make_map(size)*exp_scale)
    else:
        raise ValueError(f"Invalid map_kwarg: {map_kwarg}")
    
    weighted_ntk = jnp.multiply(ntk, weights)

    sum_sq = 0
    for i in range(1,size):
        dg = jnp.diag(weighted_ntk, -i)
        sum_sq += jnp.sum(jnp.square(dg))


    rescaled = 2*sum_sq/size
    return jnp.squeeze( rescaled / (size*jnp.sum(jnp.square(jnp.diag(ntk)))+1))

#
# def generate_tridiagonal(n: int, a: float, b: float, c: float) -> jnp.ndarray:
#     """Generate an n x n tridiagonal matrix with a on the main diagonal,
#     b on the subdiagonal, and c on the superdiagonal."""
#     main_diag = jnp.full(n, a)
#     sub_diag = jnp.full(n - 1, b)
#     super_diag = jnp.full(n - 1, c)
#     return jnp.diag(main_diag) + jnp.diag(sub_diag, -1) + jnp.diag(super_diag, 1)
#
#
# def generate_tridiagonal_bis(n: int, a: float, b: float, c: float) -> jnp.ndarray:
#     """Generate an n x n tridiagonal matrix with a on the main diagonal,
#     b on the subdiagonal, and c on the superdiagonal."""
#     main_diag = jnp.full(n, a)
#     sub_diag = jnp.full(n - 2, b)
#     # super_diag = jnp.full(n - 1, c)
#     super_diag = jnp.full(n - 1, 10)
#     return jnp.diag(main_diag) + jnp.diag(sub_diag, -2) + jnp.diag(super_diag, 1)
#
#
#
# if __name__ == "__main__":
#
#     n = 5
#     a = 40.0
#     b = 10.0
#     c = 0.0
#     X = generate_tridiagonal(n, a, b, c)
#     print(f"{X=}")
#     print(measure_of_diagonal_strength(X, 0))
#
#     X = jnp.eye(n)
#     print(f"{X=}")
#     print(measure_of_diagonal_strength(X, 0))
#
#     print(f"{X=}")
#     print(measure_of_diagonal_strength(X, 0))
#
#     X = jnp.eye(n).at[4,0].set(12.0)
#     print(f"{X=}")
#     print(measure_of_diagonal_strength(X, 0))
#
#     X_bis = jnp.eye(n).at[4,3].set(12.0)
#     print(f"{X_bis=}")
#     print(measure_of_diagonal_strength(X_bis, 0))



