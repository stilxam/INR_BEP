from typing import Any, Callable, Generator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from common_dl_utils.config_creation import Config
import common_jax_utils as cju



def make_init_apply(config: Config, key_gen: Generator) -> Tuple[Callable, Callable]:
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


