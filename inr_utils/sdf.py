import jax
from equinox import Module
import equinox as eqx
import jax.numpy as jnp

import trimesh
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from images import make_lin_grid
from skimage.measure import marching_cubes
import pymeshlab


class occupancySDF(Module):
    mesh_path:Path
    mesh:trimesh.Trimesh

    """
    Target function for the signed distance function of a mesh
    """
    def __init__(self, mesh_name:str, keep_aspect_ratio: bool = True):
        self.mesh_path = Path.cwd().joinpath("example_data", f"{mesh_name}.ply")
        self.mesh = trimesh.load(self.mesh_path)
            
        if not self.mesh.is_watertight:
            raise ValueError("Mesh is not watertight, please fix the mesh before using this class")
        
        # Center and normalize the mesh vertices
        vertices = self.mesh.vertices
        vertices -= np.mean(vertices, axis=0, keepdims=True)
        
        if keep_aspect_ratio:
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
            raise ValueError("Rescaled mesh is not watertight, please fix the mesh before using this class")

    def __call__(self, coords:jax.Array) -> jax.Array:
        """
        Evaluates the mesh as an occupancy function

        :parameter coords: the coordinates to evaluate the occupancy function at shape (N, 3)
        :return: the occupancy function values at the coordinates shape (N, 1)
        """
        return jax.numpy.array(self.mesh.contains(coords))
    
    def get_sdf(self, coords:jax.Array) -> jax.Array:
        """
        Evaluates the mesh as an SDF function

        :parameter coords: the coordinates to evaluate the SDF function at shape (N, 3)
        :return: the SDF function values at the coordinates shape (N, 1)
        """
        return jax.numpy.array(trimesh.proximity.signed_distance(self.mesh, coords))

if __name__ == "__main__":
    mesh_path = "example_data/Armadillo.ply"
    sdf = occupancySDF(mesh_path)

    print(sdf(jax.numpy.array([[0, 0.5, 0]])))