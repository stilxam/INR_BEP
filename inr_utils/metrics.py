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
from typing import Callable, Union, Tuple, Optional
import tempfile
from io import BytesIO

import jax
from jax import numpy as jnp
import equinox as eqx
import PIL
import numpy as np
import librosa
import plotly.graph_objects as go
import skimage
import trimesh

from common_dl_utils.metrics import Metric, MetricFrequency, MetricCollector  # noqa
from inr_utils.images import scaled_array_to_image, evaluate_on_grid_batch_wise, evaluate_on_grid_vmapped, \
    make_lin_grid, make_gif
from inr_utils.losses import mse_loss
from inr_utils.states import handle_state



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

        @eqx.filter_jit
        def _evaluate_by_batch(inr: eqx.Module, grid: jax.Array, data_index: Optional[Union[int, jax.Array]]):
            results = evaluate_on_grid_batch_wise(inr, grid, batch_size, False)
            if data_index is not None:
                target = partial(target_function, data_index=data_index)
            else:
                target = target_function
            reference = evaluate_on_grid_batch_wise(target, grid, batch_size, False)
            return mse_loss(results, reference)

        @eqx.filter_jit
        def _evaluate_at_once(inr: eqx.Module, grid: jax.Array, data_index: Optional[Union[int, jax.Array]]):
            results = evaluate_on_grid_vmapped(inr, grid)
            if data_index is not None:
                target = partial(target_function, data_index=data_index)
            else:
                target = target_function
            reference = evaluate_on_grid_vmapped(target, grid)
            return mse_loss(results, reference)

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
        mse = self._evaluate_on_grid(inr, self.grid, data_index=data_index)
        return {'MSE_on_fixed_grid': mse}


class AudioMetricsOnGrid(Metric):
    """
    Evaluate audio metrics between the INR output and target audio.
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            target_audio: np.ndarray,
            grid_size: int,
            batch_size: int = 1024,
            sr: int = 16000,
            frequency: str = 'every_n_batches',
            save_path: Optional[str] = None  # New parameter for saving audio
    ):
        """
        Args:
            target_audio: Original audio signal to compare against
            grid_size: Number of time points to evaluate
            batch_size: Size of batches for evaluation
            sr: Sampling rate of the audio
            frequency: How often to compute metrics
            save_path: Path to save reconstructed audio (optional)
        """
        self.target_audio = target_audio
        self.sr = sr
        self.save_path = save_path
        self.frequency = MetricFrequency(frequency)

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

    def _compute_snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio."""
        noise = original - reconstructed
        signal_power = np.sum(original ** 2)
        noise_power = np.sum(noise ** 2)

        if noise_power == 0:
            return float('inf')

        return 10 * np.log10(signal_power / noise_power)

    def _compute_spectral_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float]:
        """Compute spectral convergence and magnitude error."""
        orig_spec = np.abs(librosa.stft(original))
        recon_spec = np.abs(librosa.stft(reconstructed))

        # Spectral convergence
        spec_conv = np.linalg.norm(orig_spec - recon_spec, 'fro') / np.linalg.norm(orig_spec, 'fro')

        # Magnitude error
        mag_error = np.mean(np.abs(orig_spec - recon_spec))

        return spec_conv, mag_error

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
            sf.write(self.save_audio, reconstructed, self.sr)

        # Compute time domain metrics
        mse = np.mean((original - reconstructed) ** 2)
        snr = self._compute_snr(original, reconstructed)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        # Compute frequency domain metrics
        spec_conv, mag_error = self._compute_spectral_metrics(original, reconstructed)

        metrics = {
            'audio_snr': snr,
            'audio_psnr': psnr,
            'audio_mse': mse,
            'audio_spectral_convergence': spec_conv,
            'audio_magnitude_error': mag_error,
            'reconstructed_audio': reconstructed  # Add reconstructed audio to metrics
        }

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
        self.resolution = grid_resolution[0]  # Assume uniform resolution for now

        self.target_inside = target_function(self.grid_points)
        self.batch_size = batch_size

    def compute(self, **kwargs):
        inr = kwargs['inr']
        state = kwargs.get("state", None)

        inr = self.wrap_inr(inr)

        # pred_values = inr(self.grid_points)
        # pred_values = jax.vmap(inr)(self.grid_points)
        sdf_values = evaluate_on_grid_batch_wise(inr, self.grid_points, batch_size=self.batch_size, apply_jit=False)
        # Convert to occupancy grids (inside = True, outside = False)
        sdf_reshaped = sdf_values.reshape((self.resolution, self.resolution, self.resolution))

        vertices, faces, normals, values = skimage.measure.marching_cubes(sdf_reshaped, level=0.0)

        shape = trimesh.Trimesh(vertices=vertices, faces=faces)
        pred_inside = shape.contains(self.grid_points)

        # pred_inside = pred_values <= 0

        # Compute intersection and union
        intersection = np.logical_and(pred_inside, self.target_inside)
        union = np.logical_or(pred_inside, self.target_inside)

        # Calculate Jaccard index
        intersection_sum = np.sum(intersection)
        union_sum = np.sum(union)

        jaccard = 1.0 if union_sum == 0 else intersection_sum / union_sum

        return {'jaccard_index': float(jaccard)}

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
                 frequency: str = 'every_n_batches'
                 ):
        self.frequency = MetricFrequency(frequency)
        self.resolution = grid_resolution
        self.batch_size = batch_size

    # def __call__(self, *args, **kwargs) -> dict:
    def compute(self, **kwargs) -> dict:
        inr = kwargs['inr']
        inr = self.wrap_inr(inr)

        state = kwargs.get("state", None)

        grid = make_lin_grid(-1, 1, self.resolution, 3)
        grid = grid.reshape((-1, 3))

        # sdf_values = jax.vmap(inr)(grid)
        sdf_values = evaluate_on_grid_batch_wise(inr, grid, batch_size=self.batch_size, apply_jit=False)
        # batched_grid = grid.reshape((-1, self.batch_size, 3))
        # batched_sdf_values = jax.vmap(self.inr, in_axes=0)(batched_grid)
        # sdf_values = batched_sdf_values.reshape((-1, 1))
        fig = go.Figure(data=go.Isosurface(
            x=grid[:, 0],
            y=grid[:, 1],
            z=grid[:, 2],
            value=sdf_values,
            isomin=-0.1,
            isomax=0.1,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False)
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


class MarchingCube(Metric):
    """
    Reconstructs the SDF of a mesh from an INR
    """
    required_kwargs = set({'inr'})

    def __init__(self,
                 resolution: int = 100,
                 batch_size: int = 1024
                 ):
        self.frequency = MetricFrequency('every_n_batches')  # Default frequency
        self.resolution = resolution
        self.batch_size = batch_size

    # def __call__(self, *args, **kwargs) -> dict:
    def compute(self, **kwargs) -> dict:
        inr = kwargs['inr']
        inr = self.wrap_inr(inr)

        state = kwargs.get("state", None)

        grid = make_lin_grid(-1, 1, self.resolution, 3)
        grid = grid.reshape((-1, 3))

        # sdf_values = jax.vmap(inr)(grid)
        sdf_values = evaluate_on_grid_batch_wise(inr, grid, batch_size=self.batch_size, apply_jit=False)
        # batched_grid = grid.reshape((-1, self.batch_size, 3))
        # batched_sdf_values = jax.vmap(self.inr, in_axes=0)(batched_grid)
        sdf_reshaped = sdf_values.reshape((self.resolution, self.resolution, self.resolution))
        vertices, faces, normals, values = skimage.measure.marching_cubes(sdf_reshaped, level=0.0)

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


from .nerf_utils import SyntheticScenesHelper, ViewReconstructor
from skimage.metrics import structural_similarity as ssim

class ViewSynthesisComparison(Metric):
    """
    Compute the mean squared error between the target image and the rendered image from the NeRF
    """
    required_kwargs = set({'inr'})

    def __init__(
            self,
            split: str,
            name: str,
            width: int,
            height: int,
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
    ):
        self._cpu = jax.devices('cpu')[0]
        self._gpu = jax.devices('gpu')[0]
        folder = f"./synthetic_scenes/{split}/{name}"
        if not os.path.exists(folder):
            raise ValueError(f"Following folder does not exist: {folder}")

        with jax.default_device(self._cpu):
            # if we've already stored everything in a single big npz file, just load that
            target_path = f"{folder}/pre_processed.npz"
            if os.path.exists(target_path):
                pre_processed = np.load(target_path)
                self.target_images = jnp.asarray(pre_processed['images'][:])
                self.target_poses = jnp.asarray(pre_processed['poses'][:])
                self.target_ray_origins = jnp.asarray(pre_processed['ray_origins'][:])
                self.target_ray_directions = jnp.asarray(pre_processed['ray_directions'][:])
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

        self.view_reconstructor = ViewReconstructor(
            num_coarse_samples=num_coarse_samples,
            num_fine_samples=num_fine_samples,
            near=near,
            far=far,
            noise_std=noise_std,
            white_bkgd=white_bkgd,
            lindisp=lindisp,
            randomized=randomized,
            height=height,
            width=width,
            folder=folder,
            key=jax.random.PRNGKey(0)
        )


        self.frequency = MetricFrequency(frequency)
        self.width = width
        self.height = height
        self.batch_size = batch_size

    def compute(self, **kwargs) -> dict:
        """
        Compute the mean squared error between the target image and the rendered image from the NeRF
        """
        inr = kwargs['inr']
        inr = self.wrap_inr(inr)

        # Render the image
        rendered_images, rendered_depth = jax.vmap(self.view_reconstructor, in_axes=(None, 0, 0))(inr, self.target_ray_directions, self.target_ray_origins)
        # Compute the mean squared error
        mses = jax.vmap(self.mean_squared_error)(rendered_images, self.target_images)
        psnrs = jax.vmap(self.peak_signal_to_noise_ratio)(rendered_images, self.target_images)
        ssims = jax.vmap(self.structured_similarity_index)(rendered_images, self.target_images)

        # if ssim is not vmappable, we can use the following line instead
        # ssims = [self.structured_similarity_index(target_image, rendered_image) for target_image, rendered_image in zip(self.target_images, rendered_images)]

        return {'mse': mses, 'psnr': psnrs, 'ssim': ssims}


    @staticmethod
    def mean_squared_error(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(x - y))

    @staticmethod
    def peak_signal_to_noise_ratio(x: jax.Array, y: jax.Array) -> jax.Array:
        peak = jnp.max(y)
        return 10 * jnp.log10(peak ** 2 / jnp.mean(jnp.square(x - y)))

    @staticmethod
    def structured_similarity_index(x: jax.Array, y: jax.Array) -> float:
        """Compute the Structural Similarity Index (SSIM) between two images."""
        return ssim(x, y)


    @staticmethod
    def wrap_inr(inr):
        def wrapped(*args, **kwargs):
            out = inr(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            return out

        return wrapped


