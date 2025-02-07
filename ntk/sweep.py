
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from .analysis import analyze_fft, analyze_fft_spectrum, decompose_ntk, get_NTK_ntvp, measure_of_diagonal_strength
from .config import get_config, get_activation_kwargs

from .models import make_init_apply
from .visualization import plot_ntk_kernels #, plot_fft_spectrum
from common_jax_utils import key_generator
from typing import Tuple, Optional, Dict, Any

key_gen = key_generator(jax.random.PRNGKey(0))

def setup_sweep_config() -> Tuple[str, float, Optional[float], Dict[str, Any]]:
    """Initialize wandb and get configuration parameters."""
    wandb.init()
    layer_type = wandb.config.layer_type
    param1= wandb.config.param1
    param2= wandb.config.get("param2", None)

    activation_kwargs = get_activation_kwargs(layer_type, param1, param2)
    return layer_type, param1, param2, activation_kwargs


def compute_ntk(n: int, layer_type: str, activation_kwargs: dict) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Compute NTK and its eigenvalues."""
    config = get_config(layer_type, activation_kwargs)
    init_fn, apply_fn = make_init_apply(config, key_gen)
    params = init_fn()
    # flattened_locations = get_flattened_locations(n)
    flattened_locations = jnp.linspace(0.0, 1.0, n).reshape(-1, 1)

    ntvp = get_NTK_ntvp(apply_fn)
    NTK = ntvp(flattened_locations, flattened_locations, params)

    eigvals, _, _, condition_number = decompose_ntk(NTK)
    return NTK, eigvals, condition_number


def main_sweep() -> None:
    """Main sweep function."""
    # Setup configuration
    layer_type, param1, param2, activation_kwargs = setup_sweep_config()

    # Compute NTK and eigenvalues
    NTK, eigvals, condition_number = compute_ntk(n=100, layer_type=layer_type, activation_kwargs=activation_kwargs)

    lin_measure = measure_of_diagonal_strength(NTK, map_kwarg=0)
    const_measure = measure_of_diagonal_strength(NTK, map_kwarg=1)
    inv_measure = measure_of_diagonal_strength(NTK, map_kwarg=2)
    exp_measure = measure_of_diagonal_strength(NTK, map_kwarg=-1)
    ntk_fig = plot_ntk_kernels(NTK, layer_type, activation_kwargs)

    wandb.log({
        "layer_type": layer_type,
        "activation_kwargs": activation_kwargs,
        "ntk_condition_number": float(condition_number),
        # "max_eigenvalue": float(eigvals[0]),
        # "min_eigenvalue": float(eigvals[-1]),
        "eigvals": wandb.Histogram(eigvals),
        "ntk_plot": wandb.Image(ntk_fig),
        "lin_measure": float(lin_measure),
        "const_measure": float(const_measure),
        "inv_measure": float(inv_measure),
        "exp_measure": float(exp_measure),

    })
