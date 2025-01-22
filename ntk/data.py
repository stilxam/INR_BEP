import jax
import jax.numpy as jnp
from typing import Callable

from inr_utils.images import make_lin_grid
from common_jax_utils import key_generator


def get_flattened_locations(n: int) -> jax.Array:
    """Get flattened locations for NTK computation."""
    return make_lin_grid(0.0, 1.0, (n, n)).reshape(-1, 2)



def get_nerf_point(key) -> jax.Array:
    key, subkey = jax.random.split(key)
    position = jax.random.uniform(key, (3,), minval=-1.0, maxval=1.0)
    key, subkey = jax.random.split(key)
    view_dir = jax.random.uniform(key, (2,), minval=0, maxval=jnp.pi)
    return jnp.concatenate([position, view_dir])

def get_nerf_flattened(n: int) -> jax.Array:
    """Get flattened locations for NeRF computation."""
    key_gen = key_generator(jax.random.PRNGKey(0))

    keys = [next(key_gen) for _ in range(n*n)]
    nerf_points = jax.vmap(get_nerf_point)(jnp.array(keys))
    return nerf_points



def make_nerf_grid(n: int) -> jax.Array:
    """Make a grid of NeRF points."""
    coords = make_lin_grid(0.0, 1.0, (n, n, n, n, n))
    coords = coords.reshape(-1, 5)
    scaling = jnp.array([1.0, 1.0, 1.0, jnp.pi, jnp.pi])
    coords = coords * scaling
    return coords

if __name__ == "__main__":
    grid = make_nerf_grid(3)
    print(grid)
    print(grid.shape)

    # nerf_flattened = get_nerf_flattened(10)
    # print(nerf_flattened)
    # print(nerf_flattened.shape)






