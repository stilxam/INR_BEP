from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import layer_name_to_title




def plot_ntk_kernels(NTK: jnp.ndarray, layer_type: str, activation_kwargs: Dict[str, float]) -> plt.Figure:
    """Plot NTK kernels."""
    plot_dir = Path.cwd().joinpath("results", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(NTK, cmap="plasma", extent=(0, 1, 0, 1))
    fig.colorbar(cax, ax=ax)
    ax.set_title(f"NTK for {layer_name_to_title(layer_type)}\n {(activation_kwargs)}")
    plt.savefig(
        plot_dir.joinpath(
            f"ntk_{layer_name_to_title(layer_type)}_{(activation_kwargs)}.png"
        )
    )
    return fig
