""" 
This module contains some example implementations for metrics you can use when training an INR.
PlotOnGrid2D and PlotOnGrid3D are for creating images/gifs (in case you are training an INR on an image or a video).
MSEOnFixedGrid computes the MSE on a given grid of coordinates

In the same namespace there is also LossStandardDeviation from common_jax_utils.metrics, 
which tracks the standard deviation of the loss in a window of training steps.
See https://github.com/SimonKoop/common_jax_utils/blob/main/src/common_jax_utils/metrics.py
"""

from functools import partial
from typing import Callable, Union
import tempfile
from io import BytesIO

import jax
from jax import numpy as jnp
import equinox as eqx
import PIL
import numpy as np

from common_dl_utils.metrics import Metric, MetricFrequency, MetricCollector
from inr_utils.images import scaled_array_to_image, evaluate_on_grid_batch_wise, evaluate_on_grid_vmapped, make_lin_grid, make_gif
from inr_utils.losses import mse_loss
from common_jax_utils.metrics import LossStandardDeviation  # the last two are just for convenience, to have them available in the same namespace


class PlotOnGrid2D(Metric):
    """
    Evaluate the INR on a two dimensional grid and turn the result into an image.
    """
    required_kwargs = set({'inr'})

    def __init__(
            self, 
            grid:Union[jax.Array, int, tuple[int, int]], 
            batch_size:int,
            use_wandb:bool, 
            frequency:str, 
            ):
        """
        :param grid: grid on which to evaluate the inr
            if this is an integer i, an i \times i grid will be created
            if this is a tuple of integers (i, j), an i \times j grid will be created
            if this is a jax.Array, this array will be used as the grid (should be of shape (height, width, 2))
        :param batch_size: size of the batches of coordinates to process at once
            *NB* if batch_size is 0, the entire grid will be processed at once
        :param use_wandb: whether to create a wandb.Image (if False, a PIL Image is created instead)
        :param frequency: frequency for the MetricCollector
        """
        if isinstance(grid, int):
            grid = (grid, grid)
        if isinstance(grid, (tuple, list)):
            grid = jnp.stack(
                jnp.meshgrid(
                    jnp.linspace(0., 1., grid[0]),
                    jnp.linspace(0., 1., grid[1]),
                    indexing='ij'
                    ),
                axis=-1
                )
        self.grid = grid
        self._batch_size = batch_size
        self.frequency = MetricFrequency(frequency)


        @eqx.filter_jit
        def _evaluate_by_batch(inr:eqx.Module, grid:jax.Array):
            return scaled_array_to_image(evaluate_on_grid_batch_wise(inr, grid, batch_size=batch_size, apply_jit=False))
        
        self._evaluate_by_batch = staticmethod(_evaluate_by_batch)

        if batch_size:
            self._evaluate_on_grid = self._evaluate_by_batch
        else:
            self._evaluate_on_grid = self._evaluate_at_once

        if use_wandb:
            import wandb  # weights and biases
        self._Image = partial(PIL.Image.fromarray, mode='RGB') if not use_wandb else wandb.Image

    @staticmethod
    @eqx.filter_jit
    def _evaluate_at_once(inr:eqx.Module, grid:jax.Array):
        return scaled_array_to_image(jax.vmap(jax.vmap(inr))(grid))
    
    def compute(self, **kwargs):
        inr = kwargs['inr']
        result = np.asarray(self._evaluate_on_grid(inr, self.grid))
        return {'image_on_grid': self._Image(result)}
    
class PlotOnGrid3D(Metric):
    """ 
    Plot the INR on a fixed, 3-dimensional grid and turn the result into a GIF
    """
    required_kwargs = set({'inr'})

    def __init__(
            self, 
            grid:Union[jax.Array, int, tuple[int, int, int]], 
            batch_size:int,
            use_wandb:bool, 
            fps:int,
            requires_scaling:bool,
            frequency:str, 
            ):
        """
        :parameter grid: grid on which to evaluate the inr
            if this is an integer i, an i by i by i grid will be created
            if this is a tuple of integers (i, j, k), an i by j by k grid will be created
            if this is a jax.Array, this array will be used as the grid (should be of shape (depth, height, width, 3))
        :parameter batch_size: size of the batches of coordinates to process at once
            *NB* if batch_size is 0, the entire grid will be processed at once
        :parameter use_wandb: whether to create a wandb.Video
            NB currently the option for False is not implemented yet
        :parameter fps: number of frames per second for the GIF files
        :parameter requires_scaling: whether a scaling from [0, 1) to {0, ... 255} should be used
        :parameter frequency: frequency for the MetricCollector. Should be one of:
            'every_batch'
            'every_n_batches'
            'every_epoch'
            'every_n_epochs'
        """
        if isinstance(grid, int):
            grid = (grid, grid, grid)
        if isinstance(grid, (tuple, list)):
            grid = make_lin_grid(0., 1., grid)
        self.grid = grid
        self._batch_size = batch_size
        self.frequency = MetricFrequency(frequency)
        self.fps = fps

        scaling = scaled_array_to_image if requires_scaling else lambda x : x


        @eqx.filter_jit
        def _evaluate_by_batch(inr:eqx.Module, grid:jax.Array):
            return scaling(evaluate_on_grid_batch_wise(inr, grid, batch_size=batch_size, apply_jit=False))
        
        @eqx.filter_jit
        def _evaluate_at_once(inr:eqx.Module, grid:jax.Array):
            return scaling(evaluate_on_grid_vmapped(inr, grid))
        
        self._evaluate_by_batch = staticmethod(_evaluate_by_batch)
        self._evaluate_at_once = staticmethod(_evaluate_at_once)

        if batch_size:
            self._evaluate_on_grid = self._evaluate_by_batch
        else:
            self._evaluate_on_grid = self._evaluate_at_once

        if use_wandb:
            import wandb
        else:
            raise NotImplementedError("No output format has been implemented for PlotOnGrid3D other than wandb.Video")
        self._Video = (lambda x: x) if not use_wandb else partial(wandb.Video, fps=self.fps, format='gif')  # TODO think of a better option if we're not logging to wandb
    
    
    def compute(self, **kwargs):
        inr = kwargs['inr']
        result = np.asarray(self._evaluate_on_grid(inr, self.grid))
        with tempfile.NamedTemporaryFile(suffix='.gif') as tmp_file:
            make_gif(result, save_location=tmp_file.name, fps=self.fps, artist_name='PlotOnGrid3D')
            tmp_file.seek(0)
            buff = BytesIO(initial_bytes=tmp_file.read())
            buff.seek(0)
            video = self._Video(buff, format='gif')
            buff.close()
        return {'image_on_grid': video}


class MSEOnFixedGrid(Metric):
    """ 
    Evaluate the INR and the target function on a fixed grid and return the mean squared error between the two
    """
    required_kwargs = set({'inr'})
    def __init__(
            self, 
            target_function:Union[Callable, eqx.Module], 
            grid:Union[jax.Array, int, list[int]],
            batch_size:int,
            frequency:str,
            num_dims:Union[int, None]=None,
            ):
        """ 
        :parameter target_function: the target function to compare the INR to
        :parameter grid: grid on which to evaluate the inr
            if this is an integer i, an i times i times... times i grid will be created 
                (so of shape (i, ..., i, num_dims) with num_dims axes of size i).
            if this is a tuple of integers (i_0, ..., i_{n-1}), a linear grid with shape (i_0, ..., i_{n-1}, n) will be created
            if this is a jax.Array, this array will be used as the grid 
                (should be of shape (grid_dimensions..., num_channels) where the inr takes num_channels inputs)
        :parameter batch_size: size of the batches of coordinates to process at once
            *NB* if batch_size is 0, the entire grid will be processed in parallel
            if the grid is large, this may lead to out of memory errors.
        :parameter frequency: frequency for the MetricCollector. Should be one of:
            'every_batch'
            'every_n_batches'
            'every_epoch'
            'every_n_epochs'
            NB the train loop in inr_utils.training does not use epochs.
        :parameter num_dims: the number of dimensions of the grid
        """
        if isinstance(grid, int):
            if num_dims is None: 
                raise ValueError(f"If grid is specified as a single integer, num_dims nees to be specified to know the dimensionality of the grid. Got {grid=} but {num_dims=}.")
            grid = num_dims * (grid,)
        if isinstance(grid, (tuple, list)):
            grid = make_lin_grid(0., 1., size=grid)
        self.grid = grid
        self.target_function = target_function
        self._batch_size = batch_size
        self.frequency = MetricFrequency(frequency)

        @eqx.filter_jit
        def _evaluate_by_batch(inr:eqx.Module, grid:jax.Array):
            results = evaluate_on_grid_batch_wise(inr, grid, batch_size, False)
            reference = evaluate_on_grid_batch_wise(target_function, grid, batch_size, False)
            return mse_loss(results, reference)
        
        @eqx.filter_jit
        def _evaluate_at_once(inr:eqx.Module, grid:jax.Array):
            results = evaluate_on_grid_vmapped(inr, grid)
            reference = evaluate_on_grid_vmapped(target_function, grid)
            return mse_loss(results, reference)
        
        if batch_size:
            self._evaluate_on_grid = _evaluate_by_batch
        else:
            self._evaluate_on_grid = _evaluate_at_once

    def compute(self, **kwargs):
        inr = kwargs['inr']
        mse = self._evaluate_on_grid(inr, self.grid)
        return {'MSE_on_fixed_grid': mse}
