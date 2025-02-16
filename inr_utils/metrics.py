""" 
This module contains some example implementations for metrics you can use when training an INR.
PlotOnGrid2D and PlotOnGrid3D are for creating images/gifs (in case you are training an INR on an image or a video).
MSEOnFixedGrid computes the MSE on a given grid of coordinates

In the same namespace there is also LossStandardDeviation from common_jax_utils.metrics, 
which tracks the standard deviation of the loss in a window of training steps.
See https://github.com/SimonKoop/common_jax_utils/blob/main/src/common_jax_utils/metrics.py
"""
import os
from functools import partial
from typing import Callable, Union, Optional
import tempfile
from io import BytesIO
from pathlib import Path

import jax
from jax import numpy as jnp
import equinox as eqx
import PIL
import numpy as np
import librosa
import plotly.graph_objects as go
import skimage
import trimesh
import wandb
from skimage.metrics import structural_similarity as ssim

from common_dl_utils.metrics import Metric, MetricFrequency, MetricCollector  # noqa
from inr_utils.images import scaled_array_to_image, evaluate_on_grid_batch_wise, evaluate_on_grid_vmapped, \
    make_lin_grid, make_gif, ContinuousImage
from inr_utils.losses import mse_loss
from inr_utils.states import handle_state
from inr_utils.nerf_utils import SyntheticScenesHelper, ViewReconstructor
import common_jax_utils as cju


class PlotOnGrid2D(Metric):
    """
    Evaluate the INR on a two dimensional grid and turn the result into an image.
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            grid: Union[jax.Array, int, tuple[int, int]],
            batch_size: int,
            use_wandb: bool,
            frequency: str,
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
        def _evaluate_by_batch(inr: eqx.Module, grid: jax.Array):
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
    def _evaluate_at_once(inr: eqx.Module, grid: jax.Array):
        return scaled_array_to_image(jax.vmap(jax.vmap(inr))(grid))

    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)
        if state is not None:
            inr = handle_state(inr, state)
        result = np.asarray(self._evaluate_on_grid(inr, self.grid))
        return {'image_on_grid': self._Image(result)}


class PlotOnGrid3D(Metric):
    """ 
    Plot the INR on a fixed, 3-dimensional grid and turn the result into a GIF
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            grid: Union[jax.Array, int, tuple[int, int, int]],
            batch_size: int,
            use_wandb: bool,
            fps: int,
            requires_scaling: bool,
            frequency: str,
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

        scaling = scaled_array_to_image if requires_scaling else lambda x: x

        @eqx.filter_jit
        def _evaluate_by_batch(inr: eqx.Module, grid: jax.Array):
            return scaling(evaluate_on_grid_batch_wise(inr, grid, batch_size=batch_size, apply_jit=False))

        @eqx.filter_jit
        def _evaluate_at_once(inr: eqx.Module, grid: jax.Array):
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
        self._Video = (lambda x: x) if not use_wandb else partial(wandb.Video, fps=self.fps,
                                                                  format='gif')  # TODO think of a better option if we're not logging to wandb

    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)
        if state is not None:
            inr = handle_state(inr, state)
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
            target_function: Union[Callable, eqx.Module],
            grid: Union[jax.Array, int, list[int]],
            batch_size: int,
            frequency: str,
            num_dims: Union[int, None] = None,
            use_wandb: bool = True,
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
                raise ValueError(
                    f"If grid is specified as a single integer, num_dims nees to be specified to know the dimensionality of the grid. Got {grid=} but {num_dims=}.")
            grid = num_dims * (grid,)
        if isinstance(grid, (tuple, list)):
            grid = make_lin_grid(0., 1., size=grid)
        self.grid = grid
        self.target_function = target_function
        self._batch_size = batch_size
        self.frequency = MetricFrequency(frequency)

        if use_wandb:
            import wandb  # weights and biases
        self._Image = partial(PIL.Image.fromarray, mode='RGB') if not use_wandb else wandb.Image

        _image_min = jnp.min(target_function.underlying_image)
        _image_max = jnp.max(target_function.underlying_image)

        @eqx.filter_jit
        def _evaluate_by_batch(inr: eqx.Module, grid: jax.Array, data_index: Optional[Union[int, jax.Array]]):
            results = evaluate_on_grid_batch_wise(inr, grid, batch_size, False)
            # scale results
            _results_min = jnp.min(results)
            _results_max = jnp.max(results)
            results = (results - _results_min) / (_results_max - _results_min)  # scale to [0, 1]
            results = results * (_image_max - _image_min) + _image_min  # scale to actual image scale
            if data_index is not None:
                target = partial(target_function, data_index=data_index)
            else:
                target = target_function
            reference = evaluate_on_grid_batch_wise(target, grid, batch_size, False)
            return mse_loss(results, reference), results

        @eqx.filter_jit
        def _evaluate_at_once(inr: eqx.Module, grid: jax.Array, data_index: Optional[Union[int, jax.Array]]):
            results = evaluate_on_grid_vmapped(inr, grid)
            # scale results
            _results_min = jnp.min(results)
            _results_max = jnp.max(results)
            results = (results - _results_min) / (_results_max - _results_min)  # scale to [0, 1]
            results = results * (_image_max - _image_min) + _image_min  # scale to actual image scale
            if data_index is not None:
                target = partial(target_function, data_index=data_index)
            else:
                target = target_function
            reference = evaluate_on_grid_vmapped(target, grid)
            return mse_loss(results, reference), results

        if batch_size:
            self._evaluate_on_grid = _evaluate_by_batch
        else:
            self._evaluate_on_grid = _evaluate_at_once

        self.batch_size = batch_size

    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)
        data_index = kwargs.get("data_index", None)
        inr = self.wrap_inr(inr)

        # mse = self._evaluate_on_grid(inr, self.grid, data_index=data_index)
        mse, resulting_image = self._evaluate_on_grid(inr, self.grid, data_index=data_index)

        # resulting_image = evaluate_on_grid_batch_wise(inr, self.grid, self.batch_size, False)
  
        psnr = self.peak_signal_to_noise_ratio(mse, 1)
        stsidx = self.evaluate_ssim(inr, self.grid, data_index=data_index)

        return {'MSE_on_fixed_grid': mse, "PSNR": psnr, "SSIM": stsidx, 'image_on_grid':self._Image(np.asarray(resulting_image))}
    
    @staticmethod
    @jax.jit
    def peak_signal_to_noise_ratio(mse:float, peak:float) -> jax.Array:
        return 10 * (jnp.log10(peak) * 2  - jnp.log10(mse))
    
    @staticmethod
    def structured_similarity_index(x: jax.Array, y: jax.Array) -> float:
        """Compute the Structural Similarity Index (SSIM) between two images."""
        return ssim(x, y, channel_axis=-1, data_range=1.)
    
    def evaluate_ssim(self, inr: eqx.Module, grid: jax.Array, data_index: Optional[Union[int, jax.Array]]):
        results = evaluate_on_grid_batch_wise(inr, grid, self.batch_size, False)
        if data_index is not None:
            target = partial(self.target_function, data_index=data_index)
        else:
            target = self.target_function
        reference = evaluate_on_grid_batch_wise(target, grid, self.batch_size, False)
        return self.structured_similarity_index(results, reference)
    @staticmethod
    def wrap_inr(inr):
        def wrapped(*args, **kwargs):
            out = inr(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            return out

        return wrapped

    
class ImageGradMetrics(Metric):
    """ 
    Evaluate the INR and the target function on a fixed grid and return the shifted mean squared error between the two
    and show the actual image
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            target_function: ContinuousImage,
            grid: Union[jax.Array, int, list[int]],
            batch_size: int,
            frequency: str,
            num_dims: Union[int, None] = None,
            use_wandb: bool = True,
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
                raise ValueError(
                    f"If grid is specified as a single integer, num_dims nees to be specified to know the dimensionality of the grid. Got {grid=} but {num_dims=}.")
            grid = num_dims * (grid,)
        if isinstance(grid, (tuple, list)):
            grid = make_lin_grid(0., 1., size=grid)
        self.grid = grid
        self.target_function = target_function
        self._batch_size = batch_size
        self.frequency = MetricFrequency(frequency)

        if use_wandb:
            import wandb  # weights and biases
        self._Image = partial(PIL.Image.fromarray, mode='RGB') if not use_wandb else wandb.Image

        _image_min = jnp.min(target_function.underlying_image)
        _image_max = jnp.max(target_function.underlying_image)

        @eqx.filter_jit
        def _evaluate_by_batch(inr: eqx.Module, grid: jax.Array, data_index: Optional[Union[int, jax.Array]]):
            results = evaluate_on_grid_batch_wise(inr, grid, batch_size, False)
            # scale results
            _results_min = jnp.min(results)
            _results_max = jnp.max(results)
            results = (results - _results_min) / (_results_max - _results_min)  # scale to [0, 1]
            results = results * (_image_max - _image_min) + _image_min  # scale to actual image scale
            if data_index is not None:
                target = partial(target_function, data_index=data_index)
            else:
                target = target_function
            reference = evaluate_on_grid_batch_wise(target, grid, batch_size, False)
            return mse_loss(results, reference), results

        @eqx.filter_jit
        def _evaluate_at_once(inr: eqx.Module, grid: jax.Array, data_index: Optional[Union[int, jax.Array]]):
            results = evaluate_on_grid_vmapped(inr, grid)
            # scale results
            _results_min = jnp.min(results)
            _results_max = jnp.max(results)
            results = (results - _results_min) / (_results_max - _results_min)  # scale to [0, 1]
            results = results * (_image_max - _image_min) + _image_min  # scale to actual image scale
            if data_index is not None:
                target = partial(target_function, data_index=data_index)
            else:
                target = target_function
            reference = evaluate_on_grid_vmapped(target, grid)
            return mse_loss(results, reference), results

        if batch_size:
            self._evaluate_on_grid = _evaluate_by_batch
        else:
            self._evaluate_on_grid = _evaluate_at_once

    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)
        data_index = kwargs.get("data_index", None)
        if state is not None:
            inr = handle_state(inr, state)
        mse, resulting_image = self._evaluate_on_grid(inr, self.grid, data_index=data_index)

        psnr = self.peak_signal_to_noise_ratio(mse, 1)
        stsidx = self.evaluate_ssim(inr, self.grid, data_index=data_index)

        return {'MSE_on_fixed_grid': mse, "PSNR": psnr, "SSIM": stsidx, 'image_on_grid':self._Image(np.asarray(resulting_image))}



class AudioMetricsOnGrid(Metric):
    """
    Evaluate audio metrics between the INR output and target audio.
    """
    required_kwargs = set({'inr'})
    
    def __init__(
            self,
            target_audio: np.ndarray,
            grid_size: Optional[int] = None,
            batch_size: int = 1024,
            sr: int = 16000,
            frequency: str = 'every_n_batches',
            save_path: Optional[str] = None
    ):
        """
        Args:
            target_audio: Original audio signal to compare against
            grid_size: Number of time points to evaluate. If None, uses target audio length
            batch_size: Size of batches for evaluation
            sr: Sampling rate of the audio
            frequency: How often to compute metrics
            save_path: Path to save reconstructed audio (optional)
        """
        self.frequency = MetricFrequency(frequency)
        
        # Load target audio if it's a path
        if isinstance(target_audio, (str, Path)):
            target_audio = np.load(target_audio)
        self.target_audio = target_audio
        
        # Use target audio length if grid_size is None
        if grid_size is None:
            grid_size = len(target_audio)
        
        self.sr = sr
        self.save_path = save_path
        
        # Create time grid for evaluation
        self.grid = jnp.linspace(0, 1, grid_size)

        # Adjust batch_size to divide grid_size evenly
        if batch_size > grid_size:
            batch_size = grid_size
        else:
            # Find largest factor of grid_size that's <= batch_size
            while grid_size % batch_size != 0 and batch_size > 1:
                batch_size -= 1

        self._batch_size = batch_size

        @eqx.filter_jit
        def _evaluate_by_batch(inr: eqx.Module, grid: jax.Array):
            # Ensure grid has correct shape for batching
            grid = grid.reshape(-1)[:, None]  # Make 2D array with shape (n, 1)
            return evaluate_on_grid_batch_wise(inr, grid, batch_size=self._batch_size, apply_jit=False)

        @eqx.filter_jit
        def _evaluate_at_once(inr: eqx.Module, grid: jax.Array):
            grid = grid.reshape(-1)[:, None]  # Make 2D array with shape (n, 1)
            return jax.vmap(inr)(grid)

        self._evaluate_by_batch = staticmethod(_evaluate_by_batch)
        self._evaluate_at_once = staticmethod(_evaluate_at_once)

        if batch_size > 1:
            self._evaluate_on_grid = self._evaluate_by_batch
        else:
            self._evaluate_on_grid = self._evaluate_at_once

    def _compute_audio_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """Compute comprehensive audio quality metrics."""
        
        # Ensure inputs are the right shape and type
        original = original.reshape(1, -1)
        reconstructed = reconstructed.reshape(1, -1)
        
        # Original metrics (MSE, SNR, PSNR)
        mse = np.mean((original - reconstructed) ** 2)
        
        # SNR calculation
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        
        # PSNR calculation
        max_val = np.max(np.abs(original))
        psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
        
        # SI-SNR calculation
        def compute_si_snr(x: jax.Array, y: jax.Array) -> float:
            """Scale-Invariant Signal-to-Noise Ratio"""
            # Zero-mean normalization
            x = x - jnp.mean(x)
            y = y - jnp.mean(y)
            
            # Calculate scale factor
            alpha = jnp.sum(x * y) / (jnp.sum(x * x) + 1e-8)
            
            # Calculate SI-SNR
            scaled = alpha * x
            noise = y - scaled
            si_snr = 10 * jnp.log10(
                jnp.sum(scaled * scaled) / (jnp.sum(noise * noise) + 1e-8)
            )
            return float(si_snr)

        def compute_stoi(x: np.ndarray, y: np.ndarray, fs: int = 16000) -> float:
            """Short-Time Objective Intelligibility"""
            try:
                from pystoi.stoi import stoi
                return float(stoi(x.squeeze(), y.squeeze(), fs, extended=False))
            except ImportError:
                print("pystoi not installed. STOI calculation skipped.")
                return float('nan')

        def compute_pesq(x: np.ndarray, y: np.ndarray, fs: int = 16000) -> float:
            """Perceptual Evaluation of Speech Quality"""
            try:
                from pesq import pesq
                return float(pesq(fs, x.squeeze(), y.squeeze(), 'wb'))
            except ImportError:
                print("pesq not installed. PESQ calculation skipped.")
                return float('nan')
            except Exception as e:
                print(f"PESQ calculation failed: {str(e)}")
                return float('nan')

        # Compute all metrics
        metrics = {
            'mse': float(mse),
            'snr': float(snr),
            'psnr': float(psnr),
            'si_snr': compute_si_snr(jnp.array(original.squeeze()), 
                                    jnp.array(reconstructed.squeeze())),
            'stoi': compute_stoi(original, reconstructed, self.sr),
            'pesq': compute_pesq(original, reconstructed, self.sr)
        }
        
        return metrics

    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)
        if state is not None:
            inr = handle_state(inr, state)

        # Get reconstruction
        reconstructed = np.array(self._evaluate_on_grid(inr, self.grid)).squeeze()

        # Ensure same length for comparison
        min_len = min(len(self.target_audio), len(reconstructed))
        original = self.target_audio[:min_len]
        reconstructed = reconstructed[:min_len]

        # Normalize reconstructed audio to [-1, 1] range
        reconstructed = reconstructed / np.max(np.abs(reconstructed))

        # Save reconstructed audio if path is provided
        if self.save_path:
            import soundfile as sf
            sf.write(self.save_path, reconstructed, self.sr)

        # Compute all metrics
        metrics = self._compute_audio_metrics(original, reconstructed)
        
        # Log audio to wandb
        try:
            import wandb
            metrics['reconstructed_audio'] = wandb.Audio(
                reconstructed, 
                sample_rate=self.sr,
                caption="Reconstructed Audio"
            )
            metrics['original_audio'] = wandb.Audio(
                original,
                sample_rate=self.sr,
                caption="Original Audio"
            )
        except ImportError:
            # If wandb is not available, just return the raw audio array
            metrics['reconstructed_audio'] = reconstructed
            metrics['original_audio'] = original

        return metrics



class JaccardIndexSDF(Metric):
    """
    Compute the Jaccard index (Intersection over Union) between the SDF of the INR and the SDF of the target function,
    using NumPy operations. The Jaccard index is computed by comparing the binary occupancy grids derived from the SDFs,
    where negative SDF values indicate points inside the shape.
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            target_function: eqx.Module,
            grid_resolution: Union[int, tuple[int, ...]],
            batch_size: int,
            num_dims: int = 3,  # Default to 3D for SDFs
            frequency: str = 'every_n_batches'
    ):
        """
        Args:
            target_function: The target SDF function to compare against
            grid_resolution: Resolution of evaluation grid. Either single int for uniform resolution
                           or tuple specifying resolution per dimension
            num_dims: Number of dimensions (defaults to 3 for SDFs)
        """
        self.frequency = MetricFrequency(frequency)

        # Handle grid resolution specification
        if isinstance(grid_resolution, int):
            grid_resolution = (grid_resolution,) * num_dims

        # Create evaluation grid
        grid_arrays = [np.linspace(-1, 1, res) for res in grid_resolution]
        grid_matrices = np.meshgrid(*grid_arrays, indexing='ij')
        self.grid_points = np.stack([m.reshape(-1) for m in grid_matrices], axis=-1)
        self.grid_resolution = grid_resolution  # Store as tuple

        # Precompute target inside values
        self.target_inside = target_function(self.grid_points)
        self.batch_size = batch_size


    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)

        inr = self.wrap_inr(inr)

        sdf_values = evaluate_on_grid_batch_wise(inr, self.grid_points, batch_size=self.batch_size, apply_jit=False)


        pred_inside = sdf_values <= 0

        intersection = np.logical_and(pred_inside, self.target_inside)
        union = np.logical_or(pred_inside, self.target_inside)
        intersection_sum = np.sum(intersection)
        union_sum = np.sum(union)

        jaccard = intersection_sum / union_sum if union_sum != 0 else -1.0

        return {
            'Naive Jaccard Index': jaccard,
        }

    @staticmethod
    def wrap_inr(inr):
        def wrapped(*args, **kwargs):
            out = inr(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            return out

        return wrapped


class SDFReconstructor(Metric):
    """
    Reconstructs the SDF of a mesh from an INR
    """
    required_kwargs = set({'inr'})

    def __init__(self,
                 grid_resolution: int,
                 batch_size: int,
                 frequency: str = 'every_n_batches',
                 num_dims: int = 3,
                 ):

        if isinstance(grid_resolution, int):
            grid_resolution = (grid_resolution,) * num_dims

            # Create evaluation grid
        grid_arrays = [np.linspace(-1, 1, res) for res in grid_resolution]
        grid_matrices = np.meshgrid(*grid_arrays, indexing='ij')
        self.grid_points = np.stack([m.reshape(-1) for m in grid_matrices], axis=-1)
        self.resolution = grid_resolution[0]  # Assume uniform resolution for now

        self.frequency = MetricFrequency(frequency)
        self.batch_size = batch_size

    # def __call__(self, *args, **kwargs) -> dict:
    def compute(self, **kwargs) -> dict:
        inr = kwargs['inr']
        inr = self.wrap_inr(inr)

        state = kwargs.get("state", None)

        sdf_values = evaluate_on_grid_batch_wise(inr, self.grid_points, batch_size=self.batch_size, apply_jit=False)

        vertices, faces, normals, values = skimage.measure.marching_cubes(
            np.array(sdf_values.reshape(self.resolution, self.resolution, self.resolution)),
            level=0.0)

        fig = go.Figure(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
        ))
        return {"Zero Level Set": fig}

    @staticmethod
    def wrap_inr(inr):
        def wrapped(*args, **kwargs):
            out = inr(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            return out

        return wrapped


class JaccardAndReconstructionIndex(Metric):
    """
    Compute the Jaccard index (Intersection over Union) between the SDF of the INR and the SDF of the target function,
    and reconstruct the SDF of the INR as a 3D mesh.
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            target_function: eqx.Module,
            grid_resolution: Union[int, tuple[int, ...]],
            batch_size: int,
            num_dims: int = 3,  # Default to 3D for SDFs
            frequency: str = 'every_n_batches'
    ):
        """
        Args:
            target_function: The target SDF function to compare against
            grid_resolution: Resolution of evaluation grid. Either single int for uniform resolution
                           or tuple specifying resolution per dimension
            num_dims: Number of dimensions (defaults to 3 for SDFs)
        """
        self.frequency = MetricFrequency(frequency)

        # Handle grid resolution specification
        if isinstance(grid_resolution, int):
            grid_resolution = (grid_resolution,) * num_dims

        # Create evaluation grid
        grid_arrays = [np.linspace(-1, 1, res) for res in grid_resolution]
        grid_matrices = np.meshgrid(*grid_arrays, indexing='ij')
        self.grid_points = np.stack([m.reshape(-1) for m in grid_matrices], axis=-1)
        self.grid_resolution = grid_resolution  # Store as tuple

        # Precompute target inside values
        self.target_inside = target_function(self.grid_points)
        self.batch_size = batch_size

    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)

        inr = self.wrap_inr(inr)

        sdf_values = evaluate_on_grid_batch_wise(inr, self.grid_points, batch_size=self.batch_size, apply_jit=False)


        sdf_grid = np.array(sdf_values.reshape(self.grid_resolution))
        vertices, faces, normals, values = skimage.measure.marching_cubes(
            sdf_grid,
            level=0.0
        )

        spacing = [2.0 / (res - 1) for res in self.grid_resolution]
        voxel_origin = [-1.0, -1.0, -1.0]

        # Transform vertices from index space to actual coordinates
        vertices = vertices * np.array(spacing)
        vertices += np.array(voxel_origin)

        # Create mesh and check containment
        shape = trimesh.Trimesh(vertices=vertices, faces=faces)
        pred_inside = shape.contains(self.grid_points)


        # Create 3D mesh figure
        fig = go.Figure(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
        ))

        intersection = np.logical_and(pred_inside, self.target_inside)
        union = np.logical_or(pred_inside, self.target_inside)
        intersection_sum = np.sum(intersection)
        union_sum = np.sum(union)

        jaccard = intersection_sum / union_sum if union_sum != 0 else -1.0

        return {
            'Jaccard Index': jaccard,
            "Zero Level Set": fig
        }

    @staticmethod
    def wrap_inr(inr):
        def wrapped(*args, **kwargs):
            out = inr(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            return out

        return wrapped



class ViewSynthesisComparison(Metric):
    """
    Compute the mean squared error between the target image and the rendered image from the NeRF
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            split: str,
            name: str,
            batch_size: int,
            frequency: str,
            num_coarse_samples: int,
            num_fine_samples: int,
            near: float,
            far: float,
            noise_std: Optional[float],
            white_bkgd: bool,
            lindisp: bool,
            randomized: bool,
            key: jax.Array,
            subset_size:Optional[int],
    ):
        self._cpu = jax.devices('cpu')[0]
        self._gpu = jax.devices('gpu')[0]
        folder = f"example_data/synthetic_scenes/{split}/{name}"
        if not os.path.exists(folder):
            raise ValueError(f"Following folder does not exist: {folder}")

        with jax.default_device(self._cpu):
            # if we've already stored everything in a single big npz file, just load that
            target_path = f"{folder}/pre_processed.npz"
            if os.path.exists(target_path):
                pre_processed = np.load(target_path)
                self.target_images = jnp.asarray(pre_processed['images'][:])  # (num_images, height, width, 3)
                self.target_poses = jnp.asarray(pre_processed['poses'][:])  # (num_images, 4, 4)
                self.target_ray_origins = jnp.asarray(pre_processed['ray_origins'][:])  # (num_images, 3)
                self.target_ray_directions = jnp.asarray(pre_processed['ray_directions'][:])  # (num_images, height* width, 3)
            else:  # otherwise, create said npz file
                print(f"creating npz archive for {split}, {name}.")
                images, poses, ray_origins, ray_directions = SyntheticScenesHelper.create_numpy_arrays(folder)
                self.target_images = jnp.asarray(images[:])
                self.target_poses = jnp.asarray(poses[:])
                self.target_ray_origins = jnp.asarray(ray_origins[:])
                self.target_ray_directions = jnp.asarray(ray_directions[:])
                np.savez(target_path, images=images, poses=poses, ray_origins=ray_origins,
                         ray_directions=ray_directions)
                print(f"    finished creating {target_path}")
        image_size = self.target_images.shape[1:]

        self.key_gen = cju.key_generator(key)

        subset_key = next(self.key_gen)
        with jax.default_device(self._cpu):
            if subset_size is not None:
                self.indices = jax.random.choice(key, self.target_images.shape[0], shape=(subset_size,), replace=False)
            else: 
                self.indices = range(self.target_images.shape[0])

        self.view_reconstructor = eqx.filter_jit(ViewReconstructor(
            num_coarse_samples=num_coarse_samples,
            num_fine_samples=num_fine_samples,
            near=near,
            far=far,
            noise_std=noise_std,
            white_bkgd=white_bkgd,
            lindisp=lindisp,
            randomized=randomized,
            batch_size=batch_size,
            default_height=image_size[0],
            default_width=image_size[1],
            folder=folder,
            #key=next(key_gen)
        ))


        self.frequency = MetricFrequency(frequency)
        self.width = image_size[0]
        self.height = image_size[1]
        #self.batch_size = batch_size

    def compute(self, **kwargs) -> dict:
        """
        Compute the mean squared error between the target image and the rendered image from the NeRF
        """
        inr = kwargs['inr']
        #inr = self.wrap_inr(inr)  # so it never returns state  # don't NeRF returns a single dict always
        state = kwargs.get("state", None)

        # Render the image

        # self.target_images  (num_images, height, width, 3)
        # self.target_ray_origins  (num_images, 3)
        # self.target_ray_directions (num_images, height, width, 3)
        mses = []
        psnrs = []
        ssims = []
        predicted_images = []


        for i in self.indices:
            target_image = jax.device_put(self.target_images[i], self._gpu)
            ray_origin = jax.device_put(self.target_ray_origins[i], self._gpu)
            ray_directions = jax.device_put(self.target_ray_directions[i], self._gpu)

            predicted_image, _ = self.view_reconstructor(
                nerf=inr,
                ray_origin=ray_origin,
                ray_directions=ray_directions,
                key=next(self.key_gen),
                state=state,
            )

            mse = self.mean_squared_error(predicted_image, target_image)
            psnr = self.peak_signal_to_noise_ratio(mse, 1.)
            ssim = self.structured_similarity_index(predicted_image, target_image)

            mses.append(np.asarray(mse))
            psnrs.append(np.asarray(psnr))
            ssims.append(np.asarray(ssim))
            predicted_images.append(wandb.Image(np.asarray(predicted_image)))

        mean_mse = np.mean(mses)
        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)

        report = {
            "mean_mse": mean_mse,
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim
        }
        report.update({
            f"reconstruction_{int(i)}": predicted_image
            for i, predicted_image in zip(self.indices, predicted_images)
        })
        return report

    @staticmethod
    @jax.jit
    def mean_squared_error(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(x - y))


    @staticmethod
    @jax.jit
    def peak_signal_to_noise_ratio(mse:float, peak:float) -> jax.Array:
        return 10 * (jnp.log10(peak) * 2  - jnp.log10(mse))

    @staticmethod
    def structured_similarity_index(x: jax.Array, y: jax.Array) -> float:
        """Compute the Structural Similarity Index (SSIM) between two images."""
        return ssim(x, y, channel_axis=-1, data_range=1.)
