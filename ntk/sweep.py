import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import wandb

from inr_utils.images import make_lin_grid
from .analysis import analyze_fft, analyze_fft_spectrum, decompose_ntk, get_NTK_ntvp, measure_of_diagonal_strength
from .config import get_config, get_activation_kwargs

from .models import make_init_apply
from .visualization import plot_ntk_kernels  #, plot_fft_spectrum
from common_jax_utils import key_generator
from typing import Tuple, Optional, Dict, Any

key_gen = key_generator(jax.random.PRNGKey(0))


def setup_sweep_config() -> Tuple[str, float, Optional[float], Dict[str, Any]]:
    """Initialize wandb and get configuration parameters."""
    wandb.init()
    layer_type = wandb.config.layer_type
    param1 = wandb.config.get("w0", None)
    param2 = wandb.config.get("s0", None)

    activation_kwargs = get_activation_kwargs(layer_type, param1, param2)
    return layer_type, param1, param2, activation_kwargs


def tree_inner_product(tree_1, tree_2):
    component_wise = jax.tree.map(lambda x, y: jnp.sum(x * y), tree_1, tree_2)
    return sum(jax.tree.leaves(component_wise))


def ntk_single(inr, loc1, loc2):
    def apply_inr(inr, location):
        return inr(location)

    inr_grad = eqx.filter_grad(apply_inr)

    return tree_inner_product(
        inr_grad(inr, loc1),
        inr_grad(inr, loc2)
    )


@eqx.filter_jit
def _ntk_single(inr, loc1loc2):
    channels = loc1loc2.shape[-1] // 2
    loc1 = loc1loc2[:channels]
    loc2 = loc1loc2[channels:]
    return ntk_single(inr, loc1, loc2)


# def naive_ntk(inr, locs):
#     locs = make_lin_grid(0, 1, n, 2)
#     channels = 1
#     batch_size = 10
#     batch_loc1_loc_2 = locs.reshape(-1, batch_size, 2 * channels)
#     n_batches = batch_loc1_loc_2.shape[0]
#     apply_single_batch = lambda batch: jax.vmap(_ntk_single, (None, 0))(inr, batch)
#
#     resulting_batches = jax.lax.map(apply_single_batch, batch_loc1_loc_2)
#     flat_ntks = resulting_batches.reshape(n_batches * batch_size)
#
#     NTK = flat_ntks.reshape(n, n).clip(min=0 + 1e-10)

def compute_ntk(n: int, layer_type: str, activation_kwargs: dict) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Compute NTK and its eigenvalues."""
    config = get_config(layer_type, activation_kwargs)
    init_fn, apply_fn, inr = make_init_apply(config, key_gen)
    params = init_fn()


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
        "ntk_condition_number": jnp.log(condition_number + 1e-5),
        "eigvals": eigvals,
        "ntk_plot": wandb.Image(ntk_fig),
        "lin_measure": float(lin_measure),
        "const_measure": float(const_measure),
        "inv_measure": float(inv_measure),
        "exp_measure": float(exp_measure),

    })
