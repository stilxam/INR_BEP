""" 
For many tasks, we train INRs by sampling coordinates to evaluate the INR and target function at, and then using the loss between the two to obtain gradients.
This module contains a Sampler class and subclasses used for sampling locations at which to evaluate INRs during training.
Note that the samplers need to be jittable in order for the train step to be jittable.
    When jitting directly, it may be necessary to use equinox.filter_jit
Also note that if a sampler is to be used to train a hypernetwork, it should be vmap-able too.
"""
import abc
from typing import Union, Optional
import os

import numpy as np


import jax
from jax import numpy as jnp
from equinox import Module
from jaxtyping import PyTree

from common_dl_utils.type_registry import register_type
from inr_utils.images import make_lin_grid
from inr_utils.nerf_utils import SyntheticScenesHelper

from pathlib import Path


@register_type
class Sampler(Module):
    """Abstract base class for Sampler classes"""

    @abc.abstractmethod

    def __call__(self, key:jax.Array)->PyTree:

        """ 
        Sample coordinates from the distribution using the prng key provided.
        :param key: jax.Array functioning as a prng key
        :return: a PyTree with relevant information for the evaluation of the INR and loss calculation.
            e.g. an array of coordinates
        """
        pass


class UniformSampler(Sampler):  # just uniform distribution in [0, 1]^num_dims
    """ 
    Samples coordinates from a uniform distribution on [0, 1)^num_dims

    :parameter batch_size: the number of coordinates to generate
    :parameter num_dims: the number of dimensions of the coordinates
    """
    batch_size: int
    num_dims: int

    def __call__(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(key, shape=(self.batch_size, self.num_dims))


class GridSampler(Sampler):
    """ 
    Sampler that creates a batch of coordinates that all lie on a grid
    The grid can be fixed (in which case the sampler is not stochastic) or shifted each time in a random direction
    """
    size: int
    num_dims: int
    static: bool
    dx: float
    grid: jax.Array
    _shape: tuple[int, ...]

    def __init__(self, size: int, num_dims: int, static: bool):
        """
        :param size: the number of points in each dimension of the grid
        :param num_dims: the number of dimensions of the grid
        :param static: whether the grid should be fixed or shifted in a random direction each time
        NB the resulting batches of samples will have shape (size**num_dims, num_dims)
        """
        self.size = size
        self.num_dims = num_dims
        self.static = static

        self.dx = 1./size
        self.grid = jnp.stack(jnp.meshgrid(*(num_dims*[jnp.linspace(0., 1.-self.dx, size)]), indexing='ij'), axis=-1)
        self._shape = num_dims*(1,)+(num_dims,)
        
    def __call__(self, key:jax.Array)->jax.Array:

        if self.static:
            return self.grid.reshape((-1, self.num_dims))
        shift = jax.random.uniform(key=key, shape=self._shape, minval=0., maxval=self.dx)
        shifted_grid = self.grid + shift
        return shifted_grid.reshape((-1, self.num_dims))


class GridSubsetSampler(Sampler):
    """ 
    Sampler that samples random subsets of a fixed size of a given set of coordinates
    """
    coordinate_set: jax.Array
    batch_size: int
    replace: bool

    def __init__(
            self,
            size: Union[int, list[int], tuple[int, ...]],
            batch_size: int,
            allow_duplicates: bool,
            min: Union[float, list[float], tuple[float, ...]] = 0.,
            max: Union[float, list[float], tuple[float, ...]] = 1.,
            num_dimensions: Union[int, None] = None,
            indexing: str = 'ij'
    ):
        """ 
        :parameter size: the size of the coordinate grid of which we want to use subsets.
            I.e. number of points in each dimension. If a scalar, this is the number of points in all dimensions.
        :parameter batch_size: number of coordinates to sample each time
            the resulting sample will have shape (batch_size, N)
        :parameter allow_duplicates: whether to allow for choosing the same element of the coordinate_set twice in the same batch
            This is used to determine the `replace` parameter in jax.random.choice
            This does not check for duplicates in coordinate_set
        :parameter min: minimum values of the grid. If a scalar, this is the minimum value in all dimensions.
        :parameter max: maximum values of the grid. If a scalar, this is the maximum value in all dimensions.
        :parameter num_dimensions: number of dimensions of the grid. If None, this is inferred from the shapes of min, max, and size.
        :parameter indexing: indexing of the grid. 'ij' for matrix style indexing, 'xy' for cartesian style indexing.
            default is 'ij'
        """
        coordinate_set = make_lin_grid(
            min=min,
            max=max,
            size=size,
            num_dimensions=num_dimensions,
            indexing=indexing
        )
        n_dim = coordinate_set.shape[-1]
        self.coordinate_set = coordinate_set.reshape((-1, n_dim))  # flatten, e.g. in case of a grid of coordinates
        self.batch_size = batch_size
        self.replace = not allow_duplicates

    def __call__(self, key:jax.Array)->jax.Array:
        idx = jax.random.choice(key, self.coordinate_set.shape[0], shape=(self.batch_size,), replace=self.replace)
        return self.coordinate_set[idx]




class NeRFSyntheticScenesSampler(Sampler):  # NB this stores all views of the object on the gpu. might be expensive for small gpus but should be fine for Snellius
    images: jax.Array
    poses: jax.Array
    ray_origins: jax.Array
    ray_directions: jax.Array
    batch_size: int
    poses_per_batch: int
    _pixels_per_pose: int

    def __call__(self, key:jax.Array)->tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        """ 
        :param key: jax prng key
        :returns: three arrays:
            ray origins
            ray directions
            prng_key
            ground truth
        """

        poses_key, directions_key, return_key = jax.random.split(key, 3)

        height, width = self.images.shape[1:3]
        grid_size = height * width
        dataset_size = self.images.shape[0]

        selected_camera_poses_idx = jax.random.choice(poses_key, dataset_size, shape=(self.poses_per_batch,))
        selected_directions_idx_flat = jax.random.choice(directions_key, grid_size,  shape=(self.poses_per_batch, self._pixels_per_pose))  # flat (raveled) indices into grid
        selected_camera_poses_idx = jnp.broadcast_to(selected_camera_poses_idx[:, None], (self.poses_per_batch, self._pixels_per_pose))

        selected_directions_idx_grid = jax.vmap(
            lambda flat_index: jnp.unravel_index(flat_index, shape=(height, width))
        )(selected_directions_idx_flat)  # multi indices into (height, width) grid

        selected_directions_idx = (selected_camera_poses_idx,)+ selected_directions_idx_grid #  multi indices into (dataset_size, height, width) array

        ray_origins = self.ray_origins[selected_camera_poses_idx]
        ray_directions = self.ray_directions[selected_directions_idx]
        ground_truth_pixel_values = self.images[selected_directions_idx]

        return ray_origins.reshape((self.batch_size, 3)), ray_directions.reshape(self.batch_size, 3), return_key, ground_truth_pixel_values.reshape((self.batch_size, 3))


    def __init__(self, split:str, name:str, batch_size:int, poses_per_batch:int, base_path:str="./synthetic_scenes", size_limit:int=-1):
        
        folder = f"{base_path}/{split}/{name}"
        if not os.path.exists(folder):
            raise ValueError(f"Following folder does not exist: {folder}")
        if batch_size % poses_per_batch:
            raise ValueError(f"batch_size should be divisible by poses_per_batch. Got {batch_size=} but {poses_per_batch=}  - note that {batch_size % poses_per_batch=}.")
        
        self.batch_size = batch_size
        self.poses_per_batch = poses_per_batch
        self._pixels_per_pose = batch_size // poses_per_batch

        
        # if we've already stored everything in a single big npz file, just load that
        target_path = f"{folder}/pre_processed.npz"
        if os.path.exists(target_path):
            pre_processed = np.load(target_path)
            self.images = jnp.asarray(pre_processed['images'][:size_limit])
            self.poses = jnp.asarray(pre_processed['poses'][:size_limit])
            self.ray_origins = jnp.asarray(pre_processed['ray_origins'][:size_limit])
            self.ray_directions = jnp.asarray(pre_processed['ray_directions'][:size_limit])
        else: # otherwise, create said npz file
            print(f"creating npz archive for {split}, {name}.")
            images, poses, ray_origins, ray_directions = SyntheticScenesHelper.create_numpy_arrays(folder)
            self.images = jnp.asarray(images[:size_limit])
            self.poses = jnp.asarray(poses[:size_limit])
            self.ray_origins = jnp.asarray(ray_origins[:size_limit])
            self.ray_directions = jnp.asarray(ray_directions[:size_limit])
            np.savez(target_path, images=images, poses=poses, ray_origins=ray_origins, ray_directions=ray_directions)
            print(f"    finished creating {target_path}")

# class ConcatSampler(Sampler):
#     """ 
#     Create a sampler out of multiple samplers
#     Each batch generated by this sampler consists of the concatenation of the batches generated by the constituent samplers
#     """
#     samplers: list[Sampler]

#     def __init__(self, *samplers: Sampler):
#         self.samplers = list(samplers)

#     def __call__(self, key):
#         keys = jax.random.split(key, len(self.samplers))
#         return jnp.concatenate([sampler(key) for key, sampler in zip(keys, self.samplers)], axis=0)


#     def __init__(self, *samplers: Sampler):
#         """ 
#         :parameter samplers: samplers to choose from
#         """
#         self.samplers = list(samplers)

#     def __call__(self, key):
#         branch_selection_key, sampler_key = jax.random.split(key)
#         branch_index = jax.random.randint(branch_selection_key, shape=(), minval=0, maxval=len(self.samplers))
#         return jax.lax.switch(branch_index, self.samplers, sampler_key)
    
    
from common_jax_utils.decorators import load_arrays 

# class SoundSampler(Sampler):
#     """ 
#     Sample coordinates from a sound. Returns batches of time windows and corresponding pressure values
#     from a sound fragment for training INRs to represent audio signals.
#     """
#     sound_fragment: jax.Array #shape (length,)
#     fragment_length: int
#     window_size: int
#     batch_size: int
    
#     @load_arrays
#     def __init__(self, sound_fragment: jax.Array, window_size: int, batch_size: int, fragment_length: Optional[int]=None):
#         """
#         Initialize the sampler.
        
#         Args:
#             sound_fragment: Array containing the audio pressure values
#             fragment_length: Length of the sound fragment
#             window_size: Size of each sampled window
#             batch_size: Number of windows to sample in each batch
#         """
#         if fragment_length is None:
#             fragment_length = sound_fragment.shape[0]
#         self.sound_fragment = sound_fragment
#         self.fragment_length = fragment_length
#         self.window_size = window_size
#         self.batch_size = batch_size

#     def __call__(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
#         """
#         Sample batches of time windows and pressure values.
        
#         Args:
#             key: PRNG key for random sampling
            
#         Returns:
#             Tuple of:
#             - time_points: Array of shape (batch_size, window_size) containing time indices
#             - pressure_values: Array of shape (batch_size, window_size) containing pressure values
#         """
#         # Sample starting points uniformly from valid range
#         start_points = jax.random.uniform(
#             key, 
#             shape=(self.batch_size,), 
#             minval=0, 
#             maxval=self.fragment_length - self.window_size
#         )
#         start_points = jnp.floor(start_points).astype(jnp.int32)
        
#         # Create time points array - each row contains window_size sequential points
#         time_points = jnp.arange(self.window_size)[None,:] + start_points[:,None]
#         time_points = time_points / self.fragment_length  # Normalize to [0,1]
        
#         # Get corresponding pressure values using vectorized indexing
#         pressure_values = jax.vmap(lambda t: self.sound_fragment[t:t+self.window_size])(start_points)
        
#         return time_points, pressure_values
    
    

  # ... (keep all existing imports and other classes) ...

class SoundSampler(Sampler):
    """ 
    Sample coordinates from a sound. Returns batches of time windows and corresponding pressure values
    from a sound fragment for training INRs to represent audio signals.
    """
    sound_fragment: jax.Array  # shape (length,)
    fragment_length: int
    window_size: int
    batch_size: int
    
    @load_arrays
    def __init__(self, sound_fragment: jax.Array, window_size: int, batch_size: int, fragment_length: Optional[int]=None):
        """
        Initialize the sampler.
        
        Args:
            sound_fragment: Array containing the audio pressure values
            fragment_length: Length of the sound fragment
            window_size: Size of each sampled window
            batch_size: Number of windows to sample in each batch
        """
        if fragment_length is None:
            fragment_length = sound_fragment.shape[0]
        self.sound_fragment = sound_fragment
        self.fragment_length = fragment_length
        self.window_size = window_size
        self.batch_size = batch_size

    def get_window(self, start_idx):
        """
        Helper function to get a window using dynamic_slice
        
        Args:
            start_idx: Starting index for the window
            
        Returns:
            Window of pressure values of size window_size
        """
        return jax.lax.dynamic_slice(self.sound_fragment, (start_idx,), (self.window_size,))

    def __call__(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Sample batches of time windows and pressure values.
        
        Args:
            key: PRNG key for random sampling
            
        Returns:
            Tuple of:
            - time_points: Array of shape (batch_size, window_size) containing time indices
            - pressure_values: Array of shape (batch_size, window_size) containing pressure values
        """
        # Sample starting points uniformly from valid range
        start_points = jax.random.uniform(
            key, 
            shape=(self.batch_size,), 
            minval=0, 
            maxval=self.fragment_length - self.window_size
        )
        start_points = jnp.floor(start_points).astype(jnp.int32)
        
        # Create time points array - each row contains window_size sequential points
        time_points = jnp.arange(self.window_size)[None,:] + start_points[:,None]
        time_points = time_points / self.fragment_length  # Normalize to [0,1]
        
        # Get corresponding pressure values using dynamic_slice
        pressure_values = jax.vmap(self.get_window)(start_points)
        
        return time_points, pressure_values  
    


import trimesh

class SDFSampler(Sampler):
    """
    Sampler that samples random subsets of a fixed size of a given set of coordinates
    """
    coords: jax.Array
    normals: jax.Array
    sdf_name: str
    batch_size: int
    on_surface_count: int
    off_surface_count: int
    keep_aspect_ratio: bool

    def __init__(self, sdf_name: str, batch_size: int, keep_aspect_ratio: bool):
        """
        :param sdf_name: name of the file without extension (must be .xyz or .ply)
        :param batch_size: size of the batch
        :param keep_aspect_ratio: if True, keep the aspect ratio of the point cloud
        """
        self.sdf_name = sdf_name
        self.batch_size = batch_size
        self.on_surface_count = batch_size // 2
        self.off_surface_count = batch_size - self.on_surface_count
        self.keep_aspect_ratio = keep_aspect_ratio
        self.load_shape()


    def load_shape(self):
        """
        Load .ply file
        """
        fp_ply = Path.cwd().joinpath("example_data", f"{self.sdf_name}.ply")
        # fp_xyz = Path.cwd().joinpath("example_data", f"{self.sdf_name}.xyz")

        mesh = trimesh.load(str(fp_ply))
        
        if not mesh.is_watertight:
            raise ValueError(f"Mesh {self.sdf_name} is not watertight")
        
        
        vertices = mesh.vertices

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
        mesh.vertices = vertices

        if not mesh.is_watertight:
            raise ValueError(f"Rescaled mesh {self.sdf_name} is not watertight, please fix the mesh before using this class")
        

        self.coords = np.array(mesh.vertices)
        self.normals = np.array(mesh.vertex_normals)

    def __call__(self, key):
        """
        Loads the normalized point cloud from the file and generates batches of surface and off-surface points
        """
        idx = jax.random.choice(key, self.coords.shape[0], shape=(self.on_surface_count,), replace=False)
        off_surface_coords = jax.random.uniform(
            key, shape=(self.off_surface_count, 3), minval=-1., maxval=1.)
        off_surface_normals = jnp.ones((self.off_surface_count, 3)) * -1.

        sp_sdf = jnp.zeros((self.on_surface_count, 1))
        nsp_sdf = jnp.ones((self.off_surface_count, 1)) * -1

        coords = jnp.concatenate([self.coords[idx], off_surface_coords], axis=0)
        normals = jnp.concatenate([self.normals[idx], off_surface_normals], axis=0)
        sdf = jnp.concatenate([sp_sdf, nsp_sdf], axis=0)
        return coords, normals, sdf

