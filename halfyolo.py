import pdb
import traceback
from pathlib import Path

import jax
from jax import jit
from jax import numpy as jnp
import optax
import wandb
import equinox as eqx
from inr_utils.images import make_lin_grid

from common_dl_utils.config_creation import Config
import common_jax_utils as cju
import neural_tangents as nt
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({'font.size': 25})
# run on cpu
jax.config.update('jax_platform_name', 'cpu')

wandb.login()

key = jax.random.PRNGKey(12398)
key_gen = cju.key_generator(key)


def get_config(layer_type: str, activation_kwargs: dict):
    config = Config()
    config.architecture = './model_components'
    config.model_type = 'inr_modules.CombinedINR'
    config.model_config = Config()
    config.model_config.in_size = 2
    config.model_config.out_size = 2
    config.model_config.terms = [
        ('inr_modules.MLPINR.from_config', {
            'hidden_size': 1028,
            'num_layers': 3,
            'layer_type': layer_type,
            'num_splits': 1,
            'activation_kwargs': activation_kwargs,
            'initialization_scheme': 'initialization_schemes.siren_scheme',
            'initialization_scheme_kwargs': {'w0': 12.},
            'post_processor': 'auxiliary.real_scalar'
        })
    ]
    return config


def make_init_apply(config):
    """
    Create an initialization function and an apply function for an MLP model.

    """
    try:
        inr = cju.run_utils.get_model_from_config_and_key(
            prng_key=next(key_gen),
            config=config,
            model_sub_config_name_base='model',
            add_model_module_to_architecture_default_module=False,
        )
    except Exception as e:
        traceback.print_exc()
        print(e)
        print('\n')
        pdb.post_mortem()

    params, static = eqx.partition(inr, eqx.is_inexact_array)

    def init_fn():
        return params

    def apply_fn(_params, x):
        model = eqx.combine(_params, static)
        return model(x)

    return init_fn, jax.vmap(apply_fn, (None, 0))


def prep_x_ys(n=10):
    locations = make_lin_grid(0., 1., (n, n))
    location_dims = locations.shape
    in_channels = location_dims[-1]
    out_channels = 2
    flattened_locations = locations.reshape(-1, in_channels)

    rgb_vals = jnp.sin(2 * jnp.pi * flattened_locations)

    return flattened_locations, in_channels, rgb_vals, out_channels


def get_nkt_fns(apply_fn):
    kwargs = dict(
        f=apply_fn,
        trace_axes=(),
        vmap_axes=0
    )
    jacobian_contraction = jit(nt.empirical_ntk_fn(**kwargs, implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION))
    ntvp = jit(nt.empirical_ntk_fn(**kwargs, implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS))
    return jacobian_contraction, ntvp


def decompose_ntk(ntk):
    eigvals, eigvecs = jnp.linalg.eigh(ntk)
    rescaled_eigvals = eigvals / jnp.min(jnp.abs(eigvals))
    return eigvals, eigvecs, rescaled_eigvals


def plot_eigvals(Ks):
    n_models = len(Ks.keys())
    fig, ax = plt.subplots(n_models, 2, figsize=(25, 10 * n_models))

    for i, (model_name, K) in enumerate(Ks.items()):
        eigvals, eigvecs, rescaled_eigvals = decompose_ntk(K)
        # ax[i, 0].imshow(K, cmap='plasma')
        cax = ax[i, 0].imshow(K, cmap='plasma')
        fig.colorbar(cax, ax=ax[i, 0])
        ax[i, 0].set_title(f"{model_name.split('.')[1]} NTK")
        ax[i, 1].hist(rescaled_eigvals)
        ax[i, 1].set_xscale('log')
        ax[i, 1].set_title(f"{model_name.split('.')[1]} NTK Rescaled Eigenvalues")

    fig.suptitle("Neural Tangent Kernels and Rescaled Eigenvalues")

    plt.show()
    path = Path.cwd().joinpath("results", "plots")
    if not path.exists():
        path.mkdir(parents=True)

    plt.savefig(path.joinpath("ntk_eigvals.png"))


def main(n=10, in_channels=2, plot=True):
    setups = {
        "layer_type": [
            "inr_layers.SirenLayer",
            "inr_layers.ComplexWIRE",
            # "inr_layers.RealWIRE",
            "inr_layers.GaussianINRLayer",

        ],
        "activation_kwargs": [
            {"w0": 25.},
            {"w0": 25., "s0": 15},
            # {"w0": 25., "s0": 15},
            {"inverse_scale": 0.1},
        ],
    }

    Ks = {}

    flattened_locations = make_lin_grid(0., 1., (n, n)).reshape(-1, in_channels)

    for layer_type, activation_kwargs in zip(setups["layer_type"], setups["activation_kwargs"]):
        config = get_config(layer_type, activation_kwargs)
        init_fn, apply_fn = make_init_apply(config)
        params = init_fn()

        ntjc, ntvp = get_nkt_fns(apply_fn)

        NTK_jacobian_contraction = ntjc(flattened_locations, flattened_locations, params)
        # NTK_vector_products = ntvp(flattened_locations, flattened_locations, params)

        Ks[f"{layer_type}_{activation_kwargs}"] = NTK_jacobian_contraction

    # init_fn, apply_fn = make_init_apply(config)
    # params = init_fn()
    #
    #
    # ntjc, ntvp = get_nkt_fns(apply_fn)
    #
    # NTK_jacobian_contraction = ntjc(flattened_locations, flattened_locations, params)
    # NTK_vector_products = ntvp(flattened_locations, flattened_locations, params)

    # lin_fn = nt.linearize(apply_fn, params)
    # lin_ntjc, lin_ntvp = get_nkt_fns(lin_fn)
    # lin_NTK_jacobian_contraction = lin_ntjc(flattened_locations, flattened_locations, params)
    # lin_NTK_vector_products = lin_ntvp(flattened_locations, flattened_locations, params)

    # Ks = {
    #     "NTKJC": NTK_jacobian_contraction,
    #     "NTKVP": NTK_vector_products,
    #     "lin_NTKJC": lin_NTK_jacobian_contraction,
    #     "lin_NTKVP": lin_NTK_vector_products,
    # }

    if plot:
        plot_eigvals(Ks)


if __name__ == '__main__':
    main()
