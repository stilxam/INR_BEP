import jax
import jax.numpy as jnp

import trimesh
import numpy as np
from pathlib import Path
from common_jax_utils import key_generator


class SDFDataLoader:

    def __init__(self, sdf_name: str, batch_size: int, keep_aspect_ratio: bool, key: jax.Array):
        self._cpu = jax.devices('cpu')[0]
        self._gpu = jax.devices('gpu')[0]

        self.sdf_name = sdf_name
        self.batch_size = batch_size
        self.on_surface_count = batch_size // 2
        self.off_surface_count = batch_size - self.on_surface_count
        self.keep_aspect_ratio = keep_aspect_ratio

        self.initial_key = jax.device_put(key, self._cpu)

        fp_ply = Path.cwd().joinpath("example_data", f"{self.sdf_name}.ply")

        self.mesh = trimesh.load(str(fp_ply))

        if not self.mesh.is_watertight:
            raise ValueError(f"Mesh {self.sdf_name} is not watertight")

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

        if not self.mesh.is_watertight:
            raise ValueError(
                f"Rescaled mesh {self.sdf_name} is not watertight, please fix the mesh before using this class")

        coords = np.array(self.mesh.vertices)
        normals = np.array(self.mesh.vertex_normals)

        self.coords = jax.device_put(coords, self._cpu)
        self.normals = jax.device_put(normals, self._cpu)
        self.normals = self.normals / jnp.linalg.norm(self.normals, axis=-1, keepdims=True)

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
        off_surface_coords = jax.random.uniform(
            key, shape=(self.off_surface_count, 3), minval=-1., maxval=1.)
        off_surface_normals = jnp.ones((self.off_surface_count, 3)) * -1.

        sp_sdf = jnp.zeros((self.on_surface_count, 1))
        nsp_sdf = jnp.ones((self.off_surface_count, 1)) * -1

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


