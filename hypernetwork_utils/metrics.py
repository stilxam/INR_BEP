""" 
Module containing useful metrics for monitoring the training of hypernetwork-inr combinations
These can be used with common_dl_utils.metrics.MetricCollector
"""
from typing import Any, Union
import tempfile
from io import BytesIO
import time
from functools import partial

import jax
from jax import numpy as jnp
import equinox as eqx
import numpy as np
import PIL

from common_dl_utils.metrics import Metric, MetricFrequency, MetricCollector, add_prefix_to_dictionary_keys

from inr_utils.images import scaled_array_to_image, evaluate_on_grid_batch_wise, evaluate_on_grid_vmapped, make_lin_grid, make_gif
from inr_utils.losses import mse_loss




from common_jax_utils.metrics import LossStandardDeviation


__all__ = [
    "Metric",
    "MetricFrequency",
    "MetricCollector",
    "add_prefix_to_dictionary_keys",
    "PlotOnGrid3D",
    "MSEOnFixedGrid",
    "LossStandardDeviation"
]

class PlotOnGrid3D(Metric):
    """ 
    Evaluate the INRs obtained by evaluating a hypernetwork on a batch of datapoints on a fixed 3-dimensional grid,
    and turn the results into GIF files.
    NB this metric is rather slow to evaluate. This is not due to the evaluation of the INRs on the grid,
    but due to the creation of the GIF files.
    """
    required_kwargs = set({'hypernetwork', 'example_batch'})

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
        :parameter grid: grid on which to evaluate the INRs
            if this is an integer i, an i by i by i grid will be created
            if this is a tuple of integers (i, j, k), an i by j by k grid will be created
            if this is a jax.Array, this array will be used as the grid (should be of shape (depth, height, width, 3))
        :parameter batch_size: size of the batches of coordinates to process at once
            *NB* if batch_size is 0, the entire grid will be processed at once
            NB when choosing a batch size, take into account that the various INRs produced by the hypernetwork
                will be evaluated on the batch of coordinates in parallel
            NB this is not the size of the example_batch on which the hypernetwork is to be evaluated
        :parameter use_wandb: whether to create wandb.Video instances
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

        # implement a function for evaluating a hypernetwork on a single datapoint
        # and evaluate the resulting inr on the grid
        # this entire function can then be vmapped over a batch of datapoints
        def _single_evaluate_by_batch(hypernetwork: eqx.Module, datapoint:Any, grid:jax.Array):
            inr = hypernetwork(datapoint)
            unscaled = evaluate_on_grid_batch_wise(inr, grid, batch_size, apply_jit=False)
            return scaling(unscaled)
        
        @eqx.filter_jit
        def _evaluate_by_batch(hypernetwork: eqx.Module, example_batch: Any, grid:jax.Array):
            return jax.vmap(_single_evaluate_by_batch, in_axes=(None, 0, None))(hypernetwork, example_batch, grid)
        
        # idem but for evaluating on the entire grid at once
        def _single_evaluate_vmapped(hypernetwork: eqx.Module, datapoint:Any, grid:jax.Array):
            inr = hypernetwork(datapoint)
            unscaled = evaluate_on_grid_vmapped(inr, grid)
            return scaling(unscaled)
        
        @eqx.filter_jit
        def _evaluate_at_once(hypernetwork: eqx.Module, example_batch: Any, grid:jax.Array):
            return jax.vmap(_single_evaluate_vmapped, in_axes=(None, 0, None))(hypernetwork, example_batch, grid)
        
        self._evaluate_by_batch = _evaluate_by_batch
        self._evaluate_at_once = _evaluate_at_once

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
        hypernetwork:eqx.Module = kwargs['hypernetwork']
        example_batch = kwargs['example_batch']

        result = np.asarray(
            self._evaluate_on_grid(hypernetwork, example_batch, self.grid)
        )
        videos = {}
        t0 = time.time()
        for index, vid in enumerate(result):
            with tempfile.NamedTemporaryFile(suffix='.gif') as tmp_file:
                make_gif(vid, save_location=tmp_file.name, fps=self.fps, artist_name='PlotOnGrid3D')
                tmp_file.seek(0)
                buff = BytesIO(initial_bytes=tmp_file.read())
                buff.seek(0)
                video = self._Video(buff, format='gif')
                buff.close()
                videos[f'gif_on_grid/{index}'] = video
        t1 = time.time()
        videos["gif_on_grid/debug/video_overhead"] = t1-t0
        return videos
    
class PlotOnGrid2D(Metric):
    """ 
    Evaluate the INRs obtained by evaluating a hypernetwork on a batch of datapoints on a fixed 2-dimensional grid,
    and turn the results into images.
    """
    required_kwargs = set({'hypernetwork', 'example_batch'})

    def __init__(
            self, 
            grid:Union[jax.Array, int, tuple[int, int]], 
            batch_size:int,
            use_wandb:bool, 
            requires_scaling:bool,
            frequency:str,
            ):
        """ 
        :parameter grid: grid on which to evaluate the INRs
            if this is an integer i, an i by i grid will be created
            if this is a tuple of integers (i, j), an i by j grid will be created
            if this is a jax.Array, this array will be used as the grid (should be of shape (height, width, 2))
        :parameter batch_size: size of the batches of coordinates to process at once
            *NB* if batch_size is 0, the entire grid will be processed at once
            NB when choosing a batch size, take into account that the various INRs produced by the hypernetwork
                will be evaluated on the batch of coordinates in parallel
            NB this is not the size of the example_batch on which the hypernetwork is to be evaluated
        :parameter use_wandb: whether to create wandb.Image instances
            if False, a PIL.Image is created instead
        :parameter requires_scaling: whether a scaling from [0, 1) to {0, ... 255} should be used
        :parameter frequency: frequency for the MetricCollector. Should be one of:
            'every_batch'
            'every_n_batches'
            'every_epoch'
            'every_n_epochs'
        """
        if isinstance(grid, int):
            grid = (grid, grid)
        if isinstance(grid, (tuple, list)):
            grid = make_lin_grid(0., 1., grid)
        self.grid = grid
        self._batch_size = batch_size
        self.frequency = MetricFrequency(frequency)
        scaling = scaled_array_to_image if requires_scaling else lambda x : x

        # implement a function for evaluating a hypernetwork on a single datapoint
        # and evaluate the resulting inr on the grid
        # this entire function can then be vmapped over a batch of datapoints
        def _single_evaluate_by_batch(hypernetwork: eqx.Module, datapoint:Any, grid:jax.Array):
            inr = hypernetwork(datapoint)
            unscaled = evaluate_on_grid_batch_wise(inr, grid, batch_size, apply_jit=False)
            return scaling(unscaled)
        
        @eqx.filter_jit
        def _evaluate_by_batch(hypernetwork: eqx.Module, example_batch: Any, grid:jax.Array):
            return jax.vmap(_single_evaluate_by_batch, in_axes=(None, 0, None))(hypernetwork, example_batch, grid)
        
        # idem but for evaluating on the entire grid at once
        def _single_evaluate_vmapped(hypernetwork: eqx.Module, datapoint:Any, grid:jax.Array):
            inr = hypernetwork(datapoint)
            unscaled = evaluate_on_grid_vmapped(inr, grid)
            return scaling(unscaled)
        
        @eqx.filter_jit
        def _evaluate_at_once(hypernetwork: eqx.Module, example_batch: Any, grid:jax.Array):
            return jax.vmap(_single_evaluate_vmapped, in_axes=(None, 0, None))(hypernetwork, example_batch, grid)
        
        self._evaluate_by_batch = _evaluate_by_batch
        self._evaluate_at_once = _evaluate_at_once

        if batch_size:
            self._evaluate_on_grid = self._evaluate_by_batch
        else:
            self._evaluate_on_grid = self._evaluate_at_once

        if use_wandb:
            import wandb
        
        self._Image = partial(PIL.Image.fromarray, mode='RGB') if not use_wandb else wandb.Image
    
    def compute(self, **kwargs):
        hypernetwork:eqx.Module = kwargs['hypernetwork']
        example_batch = kwargs['example_batch']

        result = np.asarray(
            self._evaluate_on_grid(hypernetwork, example_batch, self.grid)
        )
        images = {}
        for index, img in enumerate(result):
            images[f'pictures_on_grid/{index}'] = self._Image(img)
        return images
    

class MSEOnFixedGrid(Metric):
    """ 
    Evaluate the result INRs created by a hypernetwork and the fields created by a target function on a fixed grid,
    and compute the Mean Squared Error between the two.

    This metric reports both the MSE on this grid, and the standard deviation of the MSE.
    NB this standard deviation is not the standard deviation over the grid, but the standard deviation 
    over the batch of datapoints.  
    """
    required_kwargs = set({'hypernetwork', 'example_batch'})
    def __init__(
            self, 
            target:eqx.Module, 
            grid:Union[jax.Array, int, list[int]],
            batch_size:int,
            frequency:str,
            num_dims:Union[int, None]=None,
            ):
        """ 
        :parameter target: a target function. Should be a mapping datapoint -> (location -> field_value)
        :parameter grid: grid on which to evaluate the INRs generated by the hypernetwork and the fields generated by the target function
            if this is an integer i, an i by i by i grid will be created
            if this is a tuple of integers (i, j, k), an i by j by k grid will be created
            if this is a jax.Array, this array will be used as the grid (should be of shape (depth, height, width, 3))
        :parameter batch_size: size of the batches of coordinates to process at once
            *NB* if batch_size is 0, the entire grid will be processed at once
            NB when choosing a batch size, take into account that the various INRs produced by the hypernetwork
                will be evaluated on the batch of coordinates in parallel
            NB this is not the size of the example_batch on which the hypernetwork is to be evaluated
        :parameter frequency: frequency for the MetricCollector. Should be one of:
            'every_batch'
            'every_n_batches'
            'every_epoch'
            'every_n_epochs'
        :parameter num_dims: number of dimensions of the grid. Only required if grid is specified as a single integer
        """
        if isinstance(grid, int):
            if num_dims is None: 
                raise ValueError(f"If grid is specified as a single integer, num_dims nees to be specified to know the dimensionality of the grid. Got {grid=} but {num_dims=}.")
            grid = num_dims * (grid,)
        if isinstance(grid, (tuple, list)):
            grid = make_lin_grid(0., 1., size=grid)
        self.grid = grid
        self.target_function = target
        self._batch_size = batch_size
        self.frequency = MetricFrequency(frequency)

        # implement a function for evaluating a hypernetwork and target function on a single datapoint,
        # evaluate the resulting inr and field on the grid, and return the mean squared error
        # this entire function can then be vmapped over a batch of datapoints
        def _single_evaluate_by_batch(hypernetwork: eqx.Module, data_point: Union[jax.Array, tuple[jax.Array,...], Any], grid: jax.Array):
            inr = hypernetwork(data_point)
            data_point = data_point if isinstance(data_point, (tuple, list)) else (data_point,)
            target_field = target(*data_point)

            results = evaluate_on_grid_batch_wise(inr, grid, batch_size, apply_jit=False)
            reference = evaluate_on_grid_batch_wise(target_field, grid, batch_size, apply_jit=False)
            return mse_loss(results, reference)
        
        @eqx.filter_jit
        def _evaluate_by_batch(hypernetwork: eqx.Module, example_batch: Union[jax.Array, tuple[jax.Array, ...], Any], grid:jax.Array):
            evaluator = jax.vmap(_single_evaluate_by_batch, in_axes=(None, 0, None))
            values = evaluator(hypernetwork, example_batch, grid)
            mean_value = jnp.mean(values)
            std_value = jnp.std(values)
            return mean_value, std_value
        
        # idem but for evaluating on the entire grid at once
        def _single_evaluate_at_once(hypernetwork: eqx.Module, data_point: Union[jax.Array, tuple[jax.Array,...], Any], grid: jax.Array):
            inr = hypernetwork(data_point)
            data_point = data_point if isinstance(data_point, (tuple, list)) else (data_point,)
            target_field = target(*data_point)

            results = evaluate_on_grid_vmapped(inr, grid)
            reference = evaluate_on_grid_vmapped(target_field, grid)
            return mse_loss(results, reference)
        
        @eqx.filter_jit
        def _evaluate_at_once(hypernetwork: eqx.Module, example_batch: Union[jax.Array, tuple[jax.Array, ...], Any], grid:jax.Array):
            evaluator = jax.vmap(_single_evaluate_at_once, in_axes=(None, 0, None))
            values = evaluator(hypernetwork, example_batch, grid)
            mean_value = jnp.mean(values)
            std_value = jnp.std(values)
            return mean_value, std_value
        
        if batch_size:
            self._evaluate_on_grid = _evaluate_by_batch
        else:
            self._evaluate_on_grid = _evaluate_at_once

    def compute(self, **kwargs):
        hypernetwork = kwargs['hypernetwork']
        example_batch = kwargs['example_batch']

        mean, std = self._evaluate_on_grid(hypernetwork, example_batch, self.grid)
        return {'MSE_on_fixed_grid': mean, 'MSE_on_fixed_grid_std': std}
