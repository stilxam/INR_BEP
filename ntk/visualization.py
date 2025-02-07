from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import layer_name_to_title


def plot_fft_spectrum(
    magnitude_spectrum: jnp.ndarray,
    layer_name: str,
    activation_kwargs: Dict[str, float],
) -> plt.Figure:
    """Plot FFT magnitude spectrum."""
    fig, ax = plt.subplots(figsize=(10, 10))
    size = magnitude_spectrum.shape[0]

    freqs = jnp.fft.fftfreq(size)  # Compute frequency bins
    freqs_shifted = jnp.fft.fftshift(freqs)
    cax = ax.imshow(
        magnitude_spectrum,
        cmap="viridis",
        extent=(
            freqs_shifted[0],
            freqs_shifted[-1],
            freqs_shifted[0],
            freqs_shifted[-1],
        ),
    )
    fig.colorbar(cax, ax=ax)
    ax.set_title(
        f"FFT Magnitude Spectrum\n{layer_name_to_title(layer_name)}\n{activation_kwargs}"
    )
    plot_dir = Path.cwd().joinpath("results", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        plot_dir.joinpath(
            f"fft_{layer_name_to_title(layer_name)}_{activation_kwargs}.png"
        )
    )
    return fig


def plot_ntk_kernels(NTK: jnp.ndarray, layer_type: str, activation_kwargs: Dict[str, float]) -> None:
    """Plot NTK kernels."""
    plot_dir = Path.cwd().joinpath("results", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(NTK, cmap="plasma")
    ticks = jnp.linspace(0, 1, NTK.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.colorbar(cax, ax=ax)
    ax.set_title(f"NTK for {layer_name_to_title(layer_type)}\n {activation_kwargs}")
    # plt.axis("off")
    plt.savefig(
        plot_dir.joinpath(
            f"ntk_{layer_name_to_title(layer_type)}_{activation_kwargs}.png"
        )
    )
    return fig
