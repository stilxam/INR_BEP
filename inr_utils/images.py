""" 
Module for working with images and INRs.
"""
import functools
from typing import Callable, Union, Optional

import jax
from jax import numpy as jnp
from equinox import Module
import PIL
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib import animation

def scaled_array_to_image(arr: jax.Array)->jax.Array:
    """scaled_array_to_image 
    take an array of floats in [0, 1]
    scale it to be an integer array in {0, ..., 255}

    :param arr: array to be scaled
    :return: scaled array
    """
    arr = (255*arr).astype(jnp.uint16)  # larger dtype to prevent overflow
    arr = jnp.minimum(arr, 255)
    return arr

def load_image_as_array(path:str)->jax.Array:
    """ 
    Load an image from a file and represent it as a jax array
    :parameter path: a string representing a path to a file containing an image or a numpy array
    :return: a jax.Array with contents of the image
    """
    file_extension = path.split('.')[-1]
    if file_extension == 'npy':
        return jnp.load(path)
    # it's ostensibly not just a numpy array so let's use PIL
    with PIL.Image.open(path) as pil_image:
        output = jnp.asarray(pil_image)
    return output

# =================================================================================
# the following part of this module is for creating continuous (interpolated) functions from images

def _recursive_linear_interpolation(t:jax.Array, left:jax.Array, right:jax.Array)->jax.Array:
    """recursively apply linear interpolation

    :param t: interpolation coordinates in [0, 1]. This should be an (n,) array where n is the number of dimensions
    :param left: The values of the function where t[0] == 0
    :param right: The value of the function where t[0] == 1
    :return: interpolated value

    for i0, ..., in in {0, 1}, the array arr = jnp.concatenate([left, right], axis=0) should have
    arr[i0, ..., in] = _recursive_linear_interpolation([i0, ..., in], left, right) i.e. the value of the function at t0=i0, ..., tn=in
    """
    if len(t) == 1:
        # base case: linear interpolation between two points
        return t[0]*right + (1-t[0])*left 
    t_0 = t[0]
    t_rest = t[1:]
    return (
        t_0*_recursive_linear_interpolation(t_rest, right[0], right[1])
        + (1-t_0)*_recursive_linear_interpolation(t_rest, left[0], left[1])
        )

def get_from_multi_indices(arr:Union[jax.Array, np.ndarray], multi_indices:Union[jax.Array, np.ndarray]):
    """
    if multi_indices is a multi-index into arr, e.g. [i0, i1, ..., in]
        then the result is arr[i0, i1, ..., in]
    if multi_indices is an array of multi-indices into arr
        then for a multi-index j0, j1, ..., jm into multi_indices,
        out[j0, j1, ..., jm] = arr[i0, i1, ..., in] where multi_indices[j0, j1, ..., jm] = [i0, i1, ..., in]
    
    :parameter arr: jax.Array, the elements of which we want to access through multi indices
    :parameter multi_indices: a jax.Array containing multi-indices (should have an appropriate integer type)
    :return: jax.Array as described above

    This function also works if both arr and multi_indices are numpy arrays instead of jax Arrays.
    If these types are mixed, the function should still work but cannot be vmapped or jitted.
    """
    if isinstance(arr, np.ndarray):
        return arr[tuple(np.moveaxis(multi_indices, -1, 0))]  # np.unstack only available from numpy version 2.1 on
    return arr[jnp.unstack(multi_indices, axis=-1)]
    # if len(multi_indices.shape)==1:
    #     # base case: we just have one multi_index
    #     # and want to retreive the value of arr at that multi_index
    #     return arr[tuple(multi_indices)]
    #     # n_indexed_dims = len(multi_indices)
    #     # indexed_shape = arr.shape[:n_indexed_dims]
    #     # out_shape = arr.shape[n_indexed_dims:]
    #     # arr_flat = arr.reshape((-1,)+out_shape)
    #     # return arr_flat[jnp.ravel_multi_index(multi_indices, dims=indexed_shape, mode='wrap')]  # NB for very large/high dimensional arrays, this may lead to overflow and consequently to periodic output
    # else:
    #     return jax.vmap(get_from_multi_indices, in_axes=(None, 0), out_axes=0)(arr, multi_indices)

def make_linearly_interpolated_image(image:jax.Array)->Callable[[jax.Array], jax.Array]:
    """make_interpolated_image
    Make an interpolated image from a discrete image
    This turns an (n+1)-dimensional array with final dimension size c into a function from [0, 1]^n to R^c
    which n-linearly interpolates between the values of the array

    :param image: (dim_0, ..., dim_n, c) array
    :return: function from [0, 1]^n to R^c
    """
    # assumes channels last
    image = jnp.asarray(image)
    size = jnp.asarray(image.shape[:-1], dtype=jnp.float32)-1
    n_dims = len(size)
    index_block_01 = jnp.stack(
        jnp.meshgrid(*(jnp.array([0, 1], dtype=jnp.uint32) for _ in range(n_dims)), indexing='ij'),
        axis=-1
        )  # This, in combination with get_from_multi_indices will be used to get the values of the function at the corners of the interpolation cube
    channels = image.shape[-1]
    
    return_docstring = f"""
        {n_dims}-linear interpolation of an array
        The underlying array has shape {image.shape}, where the final dimension is treated as a channel dimension.

        :param coordinates: ({n_dims},) array of interpolation coordinates in [0, 1]
        :raises ValueError: if the number of coordinates does not match the number of dimensions
        :return: ({channels=},) array of interpolated values
        """

    def interpolated_image(coordinates:jax.Array):
        if len(coordinates) != n_dims:
            raise ValueError(f"{n_dims}-linearly interpolated image requires one coordinate per non-channel dimension. Got {len(coordinates)=} but {n_dims=}.")
        lower_index, interpolation_coordinates = jnp.divmod(coordinates*size, 1)
        lower_index = lower_index.reshape(n_dims*(1,)+(n_dims,)).astype(jnp.uint32)
        indices = lower_index + index_block_01  # all indices of the corners of the interpolation cube
        values_at_indices = get_from_multi_indices(image, indices)
        return _recursive_linear_interpolation(interpolation_coordinates, values_at_indices[0], values_at_indices[1])
    
    interpolated_image.__doc__ = return_docstring
        
    return interpolated_image

def make_piece_wise_constant_interpolation(arr:jax.Array)->Callable[[jax.Array], jax.Array]:  # TODO test this function
    """ 
    Make a piece-wise continuous function out of an array
    :param arr: jax.Array of shape (dim_0, ..., dim_n, c)
    :return: a function from [0, 1]^n to R^c 
    """
    shape = jnp.array(arr.shape, dtype=jnp.float32)[:-1]
    n_dims = len(shape)

    return_docstring = f"""
        Piece-wise constant interpolation of an array. 
        The underlying array has shape {arr.shape}, where the final dimension is treated as a channel dimension

        :param coordinates: ({n_dims},) array of interpolation coordinates in [0., 1.]
        :raises ValueError: if the number of coordinates does not match the number of dimensions
        :return: ({arr.shape[-1]},) array of interpolated values
        """

    def interpolated_image(coordinates:jax.Array):
        if len(coordinates) != n_dims:
            raise ValueError(f"{n_dims=} coordinates are required but only {len(coordinates)=} coordinates were provided.")
        index = jnp.floor(coordinates * shape).astype(jnp.uint32)
        return get_from_multi_indices(arr, index)
    
    interpolated_image.__doc__ = return_docstring
    
    return interpolated_image


def scale_continuous_image_to_01(func):
    """scale_continuous_image_to_01
    given an interpolated image taking values in [0, 255], 
    scale it to take values in [0, 1]

    :param func: function (e.g. an interpolated image)
    :return: wrapped function
    """
    @functools.wraps(func)
    def scaled_image(*args, **kwargs):
        unscaled_value = func(*args, **kwargs).astype(jnp.float32)
        return unscaled_value / 255
    return scaled_image


# ==================================================================================================
# The following two classes (ContinuousImage and ArrayInterpolator) are convenience wrappers around
# the interpolation methods provided above.
# ==================================================================================================
class ContinuousImage(Module):
    """ 
    Convenience wrapper around make_interpolated_image
    """
    underlying_image: jax.Array
    continuous_image: Callable
    scaled: bool
    data_index: Optional[Union[int, jax.Array]]
    multi_image: bool = False
    interpolation_method: Callable

    def __init__(self, image:Union[jax.Array, str, list[str]], scale_to_01:bool, interpolation_method:Callable, data_index:Optional[int]):
        """ 
        :parameter image: either a path to an image or an array
            Note that this image should have the channels last
            Although we speak of images, the array can represent higher dimensional data such as videos as well.
        :parameter scale_to_01: whether to scale the image to [0, 1] using scale_continuous_image_to_01
        :parameter interpolation_method: the interpolation method to be used
            e.g. make_piece_wise_constant_interpolation or make_linearly_interpolated_image
        """
        if isinstance(image, str):
            image = load_image_as_array(image)
        elif isinstance(image, (list, tuple)):  # in order for training on multiple images at the same time to work
            image = jnp.stack([load_image_as_array(i) for i in image], axis=0) # we need to load all the arrays
            self.multi_image = True
        self.data_index = data_index # so that this (traced) jax.Array always indexes into the same big array of images.
        self.underlying_image = image
        if scale_to_01:
            self.continuous_image = scale_continuous_image_to_01(
                interpolation_method(image)
            )
            self.scaled = True
        else:
            self.continuous_image = interpolation_method(image)
            self.scaled = False
        self.interpolation_method = interpolation_method

    def __call__(self, coordinates: jax.Array, data_index:Optional[Union[int, jax.Array]]=None)->jax.Array:
        """ 
        Evaluate the continuous image at some coordinates
        """
        if self.multi_image: # the following path is not very efficient, but can be jitted to become efficient
            data_index = data_index if data_index is not None else self.data_index
            if data_index is None:
                raise ValueError("You need to specify an data_index if multiple images are provided")
            image = self.underlying_image[data_index]
            maybe_scale = scale_continuous_image_to_01 if self.scaled else lambda x: x
            continuous_image = maybe_scale(self.interpolation_method(image))
        else:
            continuous_image = self.continuous_image
            
        return continuous_image(coordinates)



class ArrayInterpolator(Module):
    """ 
    Makes functions out of `jax.Array`s
    
    :param interpolation_method: a callable that turns an array 
        into a callable that returns interpolated values in that array.
        E.G. make_piece_wise_constant_interpolation or make_linearly_interpolated_image
    :param scale_to_01: whether to scale the image to [0, 1] using scale_continuous_image_to_01
    :param channels_first: whether the "channels" dimension is in place 0 or place -1
        if True, the input array `arr` to __call__ is assumed to  be of shape (channels, ...)
        if False, it is assumed to be of shape (..., channels)
        This is relevant because the interpolation methods provided in inr_utils.images expect channels last
    """
    interpolation_method: Callable
    scale_to_01: bool
    channels_first: bool

    def __call__(self, arr:jax.Array)->Callable[[jax.Array], jax.Array]:
        """ 
        :param arr: jax.Array that is to be turned into a function from coordinates to array values
        :return: a function that takes coordinates and returns appropriate values dictated by arr and by self.interpolation_method
        """
        if self.channels_first:
            arr = jnp.moveaxis(
                arr,
                source=0,
                destination=-1  # because the interpolation methods in inr_utils.images expect channels last
            )
        interpolated_array = self.interpolation_method(arr)
        if self.scale_to_01:
            interpolated_array = scale_continuous_image_to_01(interpolated_array)
        return interpolated_array

# =================================================================================
# the next part of this module contains utils for generating images from continuous functions

def make_gif(arr:np.ndarray, save_location:str, artist_name:str, time_axis:int=0, dpi:int=150, blit:bool=True, repeat:bool=True, fps:int=4)->None:
    """ 
    Create a GIF file out of an array
    :parameter arr: array to be turned into a gif
        NB when taking one time point along the time axis, the array should have the shape (height, width, channels) or (height, width)
    :parameter save_location: where to save the gif
    :parameter artist_name: string to be passed as 'artist' metadata to the PillowWriter from matplotlib (used to create the gif)
    :parameter time_axis: axis of the array to be used as time in the gif.
    :parameter dpi: dots per inch of the gif
    :parameter blit: whether to blit the animation 
        (see matplotlib.animation.FuncAnimation for more information)
    :parameter repeat: whether to repeat the gif
    :parameter fps: frames per second of the gif
    """
    arr = np.asarray(arr)
    old_dpi = plt.rcParams['figure.dpi']
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams["figure.dpi"] = dpi
    minimum = arr.min()
    maximum = arr.max()
    norm = colors.Normalize(vmin=minimum, vmax=maximum)
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.take(arr, 0, axis=time_axis), norm=norm)

    def init():
        im.set_data(np.take(arr, 0, axis=time_axis))
        return [im]
    def animate(i):
        im.set_array(np.take(arr, i, axis=time_axis))
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=arr.shape[time_axis], blit=blit, repeat=repeat)
    writer = animation.PillowWriter(fps=fps, metadata=dict(artist=artist_name))
    anim.save(save_location, writer=writer)
    plt.close()
    plt.rcParams["figure.dpi"] = old_dpi

def make_lin_grid(
        min:Union[float, list[float], tuple[float,...]], 
        max:Union[float, list[float], tuple[float,...]], 
        size:Union[int, list[int], tuple[int,...]], 
        num_dimensions:Union[int, None]=None, 
        indexing:str='ij'
        )->jax.Array:
    """ 
    Make an array that is a linear grid.
    :parameter min: minimum values of the grid. If a scalar, this is the minimum value in all dimensions.
    :parameter max: maximum values of the grid. If a scalar, this is the maximum value in all dimensions.
    :parameter size: number of points in each dimension. If a scalar, this is the number of points in all dimensions.
    :parameter num_dimensions: number of dimensions of the grid. If None, this is inferred from the shapes of min, max, and size.
    :parameter indexing: indexing of the grid. 'ij' for matrix style indexing, 'xy' for cartesian style indexing.
        default is 'ij'
    :return: jax.Array of shape (size_0, ..., size_{n-1}, n) where n=num_dimensions
        in case of indexing 'ij', the value of the returned array at index (i_0, ..., i_{n-1}, j) is 
        min_j + i_j*(max_j-min_j)/(size_j-1)
    
        note that if for all j, min_j=0 and max_j = size_j - 1, the value at index (i_0, ..., i_{n-1}, j) is i_j
        i.e. the value at index (i_0, ..., i_{n-1}) is the index itself (but as a float dtype)
    """
    if not any(isinstance(var, (list, tuple)) for var in (min, max, size)) and num_dimensions is None:
        raise ValueError("When min, max, and size are provided as scalars, num_dimensions should be specified as an integer")
    nd = set([len(var) for var in (min, max, size) if isinstance(var, (list, tuple))] + ([num_dimensions] if num_dimensions is not None else []))
    if len(nd) != 1:
        raise ValueError(f"Got competing specifications of number of dimensions: {nd=}. Coming from {min=}, {max=}, {size=} and {num_dimensions=}.")
    nd = next(iter(nd))
    
    min = min if isinstance(min, (list, tuple)) else nd * [min]
    max = max if isinstance(max, (list, tuple)) else nd * [max]
    size = size if isinstance(size, (list, tuple)) else nd * [size]

    lin_spaces = [
        jnp.linspace(m, M, s)
        for m, M, s in zip(min, max, size)
    ]
    grid = jnp.stack(
        jnp.meshgrid(*lin_spaces, indexing=indexing),
        axis=-1
    )
    return grid

def evaluate_on_grid_batch_wise(func:Callable, grid:jax.Array, batch_size:int, apply_jit:bool=True)->jax.Array:
    """
    evaluate func on grid in a batch-wise fassion

    :param func: a function taking (c,) shaped arrays and returning (c'...) shaped arrays
    :param grid: a (spatial_dims..., c) shaped array of inputs for func
    :param batch_size: the size of each batch func should be evaluated on in parallel
    :param apply_jit: whether to use jax.jit on the vmapped version of func
    :return: a (spatial_dims..., c'...) shaped array of outcomes

    func will be vmapped over each batch
    this vmapped func is applied using jax.lax.map
    NB if you want to jit this entire function, its best to wrap it first so func, batch_size, and apply_jit are fixed
    and apply_jit should be False in this case
    """
    grid_shape = grid.shape
    batches = jnp.reshape(grid, (-1, batch_size, grid_shape[-1]))

    maybe_jit = jax.jit if apply_jit else lambda f:f

    @maybe_jit
    def evaluate_on_batch(batch):
        return jax.vmap(func)(batch)
    
    results = jax.lax.map(evaluate_on_batch, batches)
    target_shape = grid_shape[:-1] + results.shape[2:]
    return jnp.reshape(results, target_shape)

def evaluate_on_grid_vmapped(func:Callable, grid:jax.Array)->jax.Array:
    """
    evaluate func on grid in a batch wise fassion

    :param func: a function taking (c,) shaped arrays and returning (c'...) shaped arrays
    :param grid: a (spatial_dims..., c) shaped array of inputs for func
    :return: a (spatial_dims..., c'...) shaped array of outcomes

    func will be vmapped over a reshaped grid
    NB if you want to jit this function, its best to wrap it first so func is fixed
    """
    grid_shape = grid.shape
    flattened_grid = jnp.reshape(grid, (-1,  grid_shape[-1]))

    results = jax.vmap(func)(flattened_grid)
    target_shape = grid_shape[:-1] + results.shape[1:]
    return jnp.reshape(results, target_shape)
