import jax
import jax.numpy as jnp
from inr_utils.images import make_lin_grid

def get_flattened_locations(n: int) -> jax.Array:
    """Get flattened locations for NTK computation."""
    return make_lin_grid(0.0, 1.0, (n, n)).reshape(-1, 2)

