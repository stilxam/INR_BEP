"""Neural Tangent Kernel (NTK) analysis for different layer types and activation functions."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import neural_tangents as nt
import wandb
from jax import jit
from jax import numpy as jnp

import common_jax_utils as cju
from common_dl_utils.config_creation import Config
from inr_utils.images import make_lin_grid

# Configure matplotlib and JAX
plt.rcParams.update({"font.size": 25})
jax.config.update("jax_platform_name", "cpu")

# Initialize wandb and random key generator
wandb.login()
key = jax.random.PRNGKey(12398)
key_gen = cju.key_generator(key)


def get_config(layer_type: str, activation_kwargs: Dict[str, float]) -> Config:
    """Create model configuration."""
    config = Config()
    config.architecture = "./model_components"
    config.model_type = "inr_modules.CombinedINR"
    config.model_config = Config()
    config.model_config.in_size = 2
    config.model_config.out_size = 2
    config.model_config.terms = [
        (
            "inr_modules.MLPINR.from_config",
            {
                "hidden_size": 1028,
                "num_layers": 3,
                "layer_type": layer_type,
                "num_splits": 1,
                "activation_kwargs": activation_kwargs,
                "initialization_scheme": "initialization_schemes.siren_scheme",
                "initialization_scheme_kwargs": {"w0": 12.0},
                "post_processor": "auxiliary.real_scalar",
            },
        )
    ]
    return config


def make_init_apply(config: Config) -> Tuple[Callable, Callable]:
    """Create initialization and apply functions for an MLP model."""
    inr = cju.run_utils.get_model_from_config_and_key(
        prng_key=next(key_gen),
        config=config,
        model_sub_config_name_base="model",
        add_model_module_to_architecture_default_module=False,
    )

    params, static = eqx.partition(inr, eqx.is_inexact_array)

    def init_fn() -> Any:
        return params

    def apply_fn(_params: Any, x: jnp.ndarray) -> jnp.ndarray:
        model = eqx.combine(_params, static)
        return model(x)

    return init_fn, jax.vmap(apply_fn, (None, 0))


def prep_x_ys(n: int = 10) -> Tuple[jnp.ndarray, int, jnp.ndarray, int]:
    """Prepare input data."""
    locations = make_lin_grid(0.0, 1.0, (n, n))
    location_dims = locations.shape
    in_channels = location_dims[-1]
    out_channels = 2
    flattened_locations = locations.reshape(-1, in_channels)
    rgb_vals = jnp.sin(2 * jnp.pi * flattened_locations)
    return flattened_locations, in_channels, rgb_vals, out_channels


def get_nkt_fns(apply_fn: Callable) -> Tuple[Callable, Callable]:
    """Get NTK computation functions."""
    kwargs = dict(f=apply_fn, trace_axes=(), vmap_axes=0)
    jacobian_contraction = jit(
        nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION
        )
    )
    ntvp = jit(
        nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS
        )
    )
    return jacobian_contraction, ntvp


def decompose_ntk(ntk: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Decompose NTK into eigenvalues and eigenvectors."""
    eigvals, eigvecs = jnp.linalg.eigh(ntk)
    rescaled_eigvals = jnp.flipud(eigvals) / jnp.min(jnp.abs(eigvals))
    return jnp.flipud(eigvals), jnp.flipud(eigvecs.T), rescaled_eigvals


def plot_kernels(Ks: Dict[str, jnp.ndarray], names: List[str]) -> None:
    """Plot multiple kernels."""
    n_models = len(Ks.keys())
    fig, ax = plt.subplots(
        n_models, 1, figsize=(10, 10 * n_models), sharex=True, sharey=True
    )

    for i, (model_name, K) in enumerate(Ks.items()):
        cax = ax[i].imshow(K, cmap="plasma")
        fig.colorbar(cax, ax=ax[i])
        ax[i].set_title(f"{names[i]}")

    plt.axis("off")
    fig.suptitle("Empirical Neural Tangent Kernel")

    plot_dir = Path.cwd().joinpath("results", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir.joinpath("ntks.png"))
    plt.show()


def single_plot_kernels(Ks: Dict[str, jnp.ndarray], names: List[str]) -> None:
    """Plot individual kernels separately."""
    plot_dir = Path.cwd().joinpath("results", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for i, (model_name, K) in enumerate(Ks.items()):
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(K, cmap="plasma")
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"{names[i]}")
        plt.axis("off")
        fig.suptitle(f"Empirical Neural Tangent Kernel \n {names[i]}")
        plt.savefig(plot_dir.joinpath(f"ntk_{model_name}.png"))
        plt.show()


def plot_eigvals(Ks: Dict[str, jnp.ndarray], names: List[str]) -> None:
    """Plot eigenvalues of multiple kernels."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    for i, (model_name, K) in enumerate(Ks.items()):
        eigvals, eigvecs, rescaled_eigvals = decompose_ntk(K)
        ax.plot(jnp.log(rescaled_eigvals), label=f"{names[i]}", alpha=0.5)

    ax.set_title("Rescaled Eigenvalues")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("$\log(\lambda_i / \lambda_{min})$")
    ax.legend()
    fig.suptitle("Eigenvalues of NTKs")

    plot_dir = Path.cwd().joinpath("results", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir.joinpath("ntk_eigvals.png"))
    plt.show()


def analyze_fft(ntk: jnp.ndarray) -> jnp.ndarray:
    """Compute FFT of NTK."""
    fft_ntk = jnp.fft.fft2(ntk)
    return jnp.log(jnp.abs(jnp.fft.fftshift(fft_ntk)) + 2e-5)
    # return jnp.log(jnp.abs(fft_ntk) + 2e-5)


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
    return fig


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


def get_sweep_configuration() -> Dict[str, Any]:
    """Get wandb sweep configuration."""
    return {
        "method": "grid",
        "name": "ntk-layer-sweep",
        "metric": {"name": "ntk_condition_number", "goal": "minimize"},
        "parameters": {
            "layer_type": {
                "values": [
                    "inr_layers.SirenLayer",
                    "inr_layers.ComplexWIRE",
                    "inr_layers.RealWIRE",
                    "inr_layers.HoscLayer",
                    "inr_layers.SinCardLayer",
                    "inr_layers.GaussianINRLayer",
                    "inr_layers.QuadraticLayer",
                    "inr_layers.MultiQuadraticLayer",
                    "inr_layers.LaplacianLayer",
                    "inr_layers.SuperGaussianLayer",
                    "inr_layers.ExpSinLayer",
                ]
            },
            "param_scale": {"values": [0.1, 1.0, 5.0, 10.0, 17.5, 25.0, 32.5, 50]},
        },
    }


def get_activation_kwargs(layer_type: str, param_scale: float) -> Dict[str, float]:
    """Map layer type to appropriate activation kwargs."""
    if layer_type in ["inr_layers.SirenLayer"]:
        return {"w0": param_scale}
    elif layer_type in ["inr_layers.ComplexWIRE", "inr_layers.RealWIRE"]:
        return {"w0": param_scale, "s0": param_scale * 0.6}
    elif layer_type in ["inr_layers.HoscLayer", "inr_layers.SinCardLayer"]:
        return {"w0": param_scale}
    elif layer_type == "inr_layers.GaussianINRLayer":
        return {"inverse_scale": 1.0 / param_scale}
    elif layer_type == "inr_layers.SuperGaussianLayer":
        return {"a": param_scale, "b": param_scale}
    else:
        return {"a": param_scale}


def layer_name_to_title(layer_name: str) -> str:
    """Convert layer name to display title."""
    titles = {
        "inr_layers.SirenLayer": "SIREN",
        "inr_layers.ComplexWIRE": "Complex WIRE",
        "inr_layers.RealWIRE": "Real WIRE",
        "inr_layers.HoscLayer": "HOSC",
        "inr_layers.SinCardLayer": "Sine Cardinal",
        "inr_layers.GaussianINRLayer": "Gaussian Bump",
        "inr_layers.QuadraticLayer": "Quadratic",
        "inr_layers.MultiQuadraticLayer": "Multi-Quadratic",
        "inr_layers.LaplacianLayer": "Laplacian",
        "inr_layers.SuperGaussianLayer": "Super Gaussian",
        "inr_layers.ExpSinLayer": "Exponential Sine",
    }
    return titles[layer_name]


def main_sweep() -> None:
    """Main sweep function."""
    wandb.init()

    layer_type = wandb.config.layer_type
    param_scale = wandb.config.param_scale
    activation_kwargs = get_activation_kwargs(layer_type, param_scale)

    n = 10  # grid size
    in_channels = 2

    # Setup and compute NTK
    config = get_config(layer_type, activation_kwargs)
    init_fn, apply_fn = make_init_apply(config)
    params = init_fn()
    flattened_locations = make_lin_grid(0.0, 1.0, (n, n)).reshape(-1, in_channels)

    kwargs = dict(f=apply_fn, trace_axes=(), vmap_axes=0)
    ntvp = jit(
        nt.empirical_ntk_fn(
            **kwargs, implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS
        )
    )
    NTK = ntvp(flattened_locations, flattened_locations, params)

    # Compute metrics
    eigvals, _, _ = decompose_ntk(NTK)
    condition_number = jnp.abs(eigvals[0] / eigvals[-1])

    # Analyze NTK using FFT
    magnitude_spectrum = analyze_fft(NTK)
    fig = plot_fft_spectrum(magnitude_spectrum, layer_type, activation_kwargs)
    fft_metrics = analyze_fft_spectrum(magnitude_spectrum)

    plot_dir = Path.cwd().joinpath("results", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        plot_dir.joinpath(
            f"fft_{layer_name_to_title(layer_type)}_{activation_kwargs}.png"
        )
    )
    wandb.log({"fft_magnitude_spectrum": wandb.Image(fig)})
    plt.close(fig)

    # Log metrics
    wandb.log(
        {
            "layer_type": layer_type,
            "activation_kwargs": activation_kwargs,
            "ntk_condition_number": float(condition_number),
            "max_eigenvalue": float(eigvals[0]),
            "min_eigenvalue": float(eigvals[-1]),
            "eigvals": wandb.Histogram(eigvals),
            **fft_metrics,
        }
    )

    # Save NTK visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(NTK, cmap="plasma")
    fig.colorbar(cax, ax=ax)
    ax.set_title(f"NTK for {layer_name_to_title(layer_type)}\n {activation_kwargs}")
    plt.axis("off")
    plt.savefig(
        plot_dir.joinpath(
            f"ntk_{layer_name_to_title(layer_type)}_{activation_kwargs}.png"
        )
    )
    wandb.log({"ntk_plot": wandb.Image(fig)})
    plt.close()


if __name__ == "__main__":
    sweep_config = get_sweep_configuration()
    sweep_id = wandb.sweep(sweep_config, project="ntk-analysis")
    wandb.agent(sweep_id, main_sweep)
