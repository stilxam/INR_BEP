import jax
from jax import numpy as jnp
from pathlib import Path
from inr_utils.images import make_lin_grid
import matplotlib.pyplot as plt



def to_sphere(x: jax.Array):
    return jnp.array([jnp.cos(x), jnp.sin(x)])




def main(n=100):
    coords = jnp.linspace(-1, 1, n)*jnp.pi

    sphere = to_sphere(coords)
    print(sphere.shape)

    fig, ax = plt.subplots(1,1 )

    ax.scatter(sphere[0, :], sphere[1, :])
    fig.show()

if __name__ == "__main__":
    main()
