from typing import Generator

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import neural_tangents as nt
import wandb
from common_dl_utils.config_creation import Config
from inr_utils.images import make_lin_grid

from .analysis import analyze_fft, analyze_fft_spectrum, decompose_ntk, get_NTK_ntvp
from .config import get_config, get_activation_kwargs
from .data import get_flattened_locations
from .models import make_init_apply
from .visualization import plot_ntk_kernels, plot_fft_spectrum
from common_jax_utils import key_generator

key_gen = key_generator(jax.random.PRNGKey(0))

def setup_sweep_config() -> tuple[str, float, dict]:
    """Initialize wandb and get configuration parameters."""
    wandb.init()
    layer_type = wandb.config.layer_type
    param_scale = wandb.config.param_scale
    activation_kwargs = get_activation_kwargs(layer_type, param_scale)
    return layer_type, param_scale, activation_kwargs


def compute_ntk(n: int, layer_type: str, activation_kwargs: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute NTK and its eigenvalues."""
    config = get_config(layer_type, activation_kwargs)
    init_fn, apply_fn = make_init_apply(config, key_gen)
    params = init_fn()
    flattened_locations = get_flattened_locations(n)

    ntvp = get_NTK_ntvp(apply_fn)
    NTK = ntvp(flattened_locations, flattened_locations, params)
    eigvals, _, _ = decompose_ntk(NTK)
    return NTK, eigvals


def analyze_and_visualize(
    NTK: jnp.ndarray, 
    layer_type: str, 
    activation_kwargs: dict
) -> tuple[dict, plt.Figure, plt.Figure]:
    """Analyze NTK and create visualizations."""
    magnitude_spectrum = analyze_fft(NTK)
    fft_fig = plot_fft_spectrum(magnitude_spectrum, layer_type, activation_kwargs)
    fft_metrics = analyze_fft_spectrum(magnitude_spectrum)
    ntk_fig = plot_ntk_kernels(NTK, layer_type, activation_kwargs)
    return fft_metrics, fft_fig, ntk_fig


def main_sweep() -> None:
    """Main sweep function."""
    # Setup configuration
    layer_type, param_scale, activation_kwargs = setup_sweep_config()
    # Compute NTK and eigenvalues
    NTK, eigvals = compute_ntk(n=10, layer_type=layer_type, activation_kwargs=activation_kwargs)
    
    # Analyze and create visualizations
    fft_metrics, fft_fig, ntk_fig = analyze_and_visualize(NTK, layer_type, activation_kwargs)
    
    # Calculate condition number
    condition_number = jnp.abs(eigvals[0] / eigvals[-1])
    
    # Log all metrics and visualizations
    wandb.log({
        "layer_type": layer_type,
        "activation_kwargs": activation_kwargs,
        "ntk_condition_number": float(condition_number),
        "max_eigenvalue": float(eigvals[0]),
        "min_eigenvalue": float(eigvals[-1]),
        "eigvals": wandb.Histogram(eigvals),
        "fft_magnitude_spectrum": wandb.Image(fft_fig),
        "ntk_plot": wandb.Image(ntk_fig),
        **fft_metrics,
    })

    