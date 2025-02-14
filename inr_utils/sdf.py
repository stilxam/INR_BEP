import os

import jax
import jax.numpy as jnp

import trimesh
import numpy as np
from pathlib import Path
from common_jax_utils import key_generator
from typing import Union, Optional

from inr_utils.images import make_lin_grid


class SDFDataLoader:
    # Simon to Maxwell: I changed the thing with sdf_name because we might want to be able to store our data in other places than just example_data.
    def __init__(self, path: str, batch_size: int, keep_aspect_ratio: bool, grid_resolution: Union[tuple, int] = 10 , num_dims: Optional[int]=3, *, key: jax.Array):
        self._cpu = jax.devices('cpu')[0]
        self._gpu = jax.devices('gpu')[0]

        self.sdf_name = os.path.split(path)[-1].removesuffix(".ply")
        self.batch_size = batch_size
        self.on_surface_count = batch_size // 2
        self.off_surface_count = batch_size - self.on_surface_count
        self.keep_aspect_ratio = keep_aspect_ratio

        self.initial_key = jax.device_put(key, self._cpu)

        fp_ply = path#Path.cwd().joinpath("example_data", f"{self.sdf_name}.ply")

        self.mesh = trimesh.load(str(fp_ply))

        # if not self.mesh.is_watertight:
        #     raise ValueError(f"Mesh {self.sdf_name} is not watertight")

        vertices = self.mesh.vertices

        if self.keep_aspect_ratio:
            coord_max = np.amax(vertices)
            coords_min = np.amin(vertices)
        else:
            coord_max = np.amax(vertices, axis=0, keepdims=True)
            coords_min = np.amin(vertices, axis=0, keepdims=True)

        vertices = (vertices - coords_min) / (coord_max - coords_min)
        vertices -= 0.5
        vertices *= 2

        # Update the mesh vertices
        self.mesh.vertices = vertices

        # if not self.mesh.is_watertight:
        #     raise ValueError(
        #         f"Rescaled mesh {self.sdf_name} is not watertight, please fix the mesh before using this class")

        coords = np.array(self.mesh.vertices)
        normals = np.array(self.mesh.vertex_normals)

        self.coords = jax.device_put(coords, self._cpu)
        self.normals = jax.device_put(normals, self._cpu)
        self.normals = self.normals / jnp.linalg.norm(self.normals, axis=-1, keepdims=True)

        if isinstance(grid_resolution, int):
            grid_resolution = (grid_resolution,) * num_dims

        # Create evaluation grid
        grid_arrays = [np.linspace(-1, 1, res) for res in grid_resolution]
        grid_matrices = np.meshgrid(*grid_arrays, indexing='ij')
        self.grid_points = jnp.array(np.stack([m.reshape(-1) for m in grid_matrices], axis=-1))
        self.resolution = grid_resolution[0]  # Assume uniform resolution for now

        # print(f"{self.grid_points.shape=}")
        self.distances = jnp.array(trimesh.proximity.signed_distance(self.mesh, self.grid_points)) # samples, 1





    def __iter__(self):
        with jax.default_device(self._cpu):
            key_gen = key_generator(self.initial_key)

        while True:
            with jax.default_device(self._cpu):
                key = next(key_gen)

            coords, normals, sdf = self.sample(key)

            yield (
                jax.device_put(coords, self._gpu),
                jax.device_put(normals, self._gpu),
                jax.device_put(sdf, self._gpu)
            )

    @(lambda f: jax.jit(f, device=jax.devices('cpu')[0], static_argnums=0))
    def sample(self, key):
        """
        Loads the normalized point cloud from the file and generates batches of surface and off-surface points
        """
        idx = jax.random.choice(key, self.coords.shape[0], shape=(self.on_surface_count,), replace=True)
        # off_surface_coords = jax.random.uniform(
            # key, shape=(self.off_surface_count, 3), minval=-1., maxval=1.)
        idoffsurfaces = jax.random.choice(key, self.resolution**3, shape=(self.off_surface_count,), replace=True)
        off_surface_coords = self.grid_points[idoffsurfaces]


        off_surface_normals = jnp.ones((self.off_surface_count, 3)) * -1.

        sp_sdf = jnp.zeros((self.on_surface_count, 1))
        # nsp_sdf = jnp.ones((self.off_surface_count, 1)) * -1
        nsp_sdf = self.distances[idoffsurfaces].reshape(-1, 1)

        coords = jnp.concatenate([self.coords[idx], off_surface_coords], axis=0)
        normals = jnp.concatenate([self.normals[idx], off_surface_normals], axis=0)
        sdf = jnp.concatenate([sp_sdf, nsp_sdf], axis=0)
        return coords, normals, sdf

    def __call__(self, coords: jax.Array) -> jax.Array:
        """
        Evaluates the mesh as an occupancy function

        :parameter coords: the coordinates to evaluate the occupancy function at shape (N, 3)
        :return: the occupancy function values at the coordinates shape (N, 1)
        """
        return jax.numpy.asarray(self.mesh.contains(coords))

    def get_sdf(self, coords: jax.Array) -> jax.Array:
        """
        Evaluates the mesh as an SDF function

        :parameter coords: the coordinates to evaluate the SDF function at shape (N, 3)
        :return: the SDF function values at the coordinates shape (N, 1)
        """
        return jax.numpy.asarray(trimesh.proximity.signed_distance(self.mesh, coords))


if __name__ == "__main__":
    key = jax.random.PRNGKey(12398)
    key_gen = key_generator(key)
    sdf_loader = SDFDataLoader("sphere", 10, True, 10, 3, key=key)


