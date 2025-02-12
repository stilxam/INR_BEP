"""
The rendering utilities are based on the jaxnerf implementation from google-research.
"""
from typing import Optional  # ,ClassVar
import os
# import multiprocessing

import numpy as np
from PIL import Image

import jax
from jax import numpy as jnp
import equinox as eqx

# from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from common_jax_utils import key_generator

from model_components.inr_modules import NeRF


def cast_ray(z_vals, origin, direction):
    """
    Cast ray through pixel positions.
    """
    origin = origin[None, :]  # [1, 3]
    direction = direction[None, :]  # [1, 3]

    # Add batch and xyz dimensions to z_vals
    z_vals = z_vals[:, None]  # [num_samples, 1,]

    return origin + z_vals * direction


def sample_along_ray(key, origin, direction, num_samples, near, far, randomized, lindisp):
    """
    Sample along a ray.

    :param key: jnp.ndarray(float32), [2,], random number generator.
    :param origin: jnp.ndarray(float32), [3], ray origin.
    :param direction: jnp.ndarray(float32), [3], ray direction.
    :param num_samples: int, the number of samples.
    :param near: float, the distance to the near plane.
    :param far: float, the distance to the far plane.
    :param randomized: bool, use randomized samples.
    :param lindisp: bool, sample linearly in disparity rather than in depth.

    :return: z_vals: jnp.ndarray(float32), [num_samples].
    :return: coords: jnp.ndarray(float32), [num_samples, 3].


    """

    t_vals = jnp.linspace(0., 1., num_samples)
    if lindisp:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        z_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = jnp.concatenate([mids, z_vals[-1:]], -1)
        lower = jnp.concatenate([z_vals[:1], mids], -1)
        t_rand = jax.random.uniform(key, [num_samples])
        z_vals = lower + (upper - lower) * t_rand

    coords = cast_ray(z_vals, origin, direction)

    return z_vals, coords


def maybe_add_gaussian_noise(key, raw, noise_std, randomized):
    """Adds gaussian noise to `raw`, which can used to regularize it.

    :param key: jnp.ndarray(float32), [2,], random number generator.
    :param raw: jnp.ndarray(float32), arbitrary shape.
    :param noise_std: float, The standard deviation of the noise to be added. If None, no noise is added.
    :param randomized: bool, don't add noise if randomized is False.

    :return: raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
    """
    if (noise_std is not None) and randomized:
        return raw + jax.random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
    else:
        return raw


def volumetric_rendering(rgb: jax.Array, sigma: jax.Array, z_vals: jax.Array, direction: jax.Array,
                         white_bkgd: bool):  # TODO include reference to original code
    """Volumetric Rendering Function.

    :param rgb: color emited at each sample point, [num_samples, 3].
    :param sigma: density at each sample point, [num_samples, 1] or [num_samples]
    :param z_vals: scaled distance along the ray (in interval [0, 1]) [num_samples].
    :param direction: direction of the ray [3].
    :param white_bkgd: whether the background should be made white. Should be static if this function is to be jitted

    :return: comp_rgb, weights, depth
    """
    eps = 1e-10
    dists = z_vals[1:] - z_vals[:-1]  # [num_samples-1]
    dists = dists * jnp.linalg.norm(direction)

    if len(sigma.shape) == 2:
        sigma = jnp.squeeze(sigma, axis=-1)  # [num_samples] ensured

    alpha = 1.0 - jnp.exp(-sigma[:-1] * dists)  # [num_samples-1]
    accummulated_transmittance = jnp.concatenate([  # [num_samples] T_i in the NeRF paper
        jnp.ones_like(alpha[:1], alpha.dtype),
        jnp.cumprod(1.0 - alpha + eps, axis=-1)
        # TODO find out if it may be more stable to just take the exponential of the cummulative sum
    ],
        axis=-1)
    alpha = jnp.concatenate([alpha, jnp.ones_like(alpha[:1],
                                                  alpha.dtype)])  # [num_samples] essentially, treat the far end of the rendering box as opaque because the final distance should be 'infinity'
    weights = alpha * accummulated_transmittance  # [num_samples]

    resulting_rgb = (weights[:, None] * rgb).sum(axis=0)  # integration over the ray
    acc = weights.sum()
    depth = (weights * z_vals).sum(axis=-1)
    if white_bkgd:
        resulting_rgb = resulting_rgb + (1. - acc)
    return resulting_rgb, weights, depth  # for training, we only need resulting_rgb and weights, but depth can be nice for a separate visualization


def piecewise_constant_pdf(key, bins, weights, num_samples, randomized):  # TODO maybe rename
    """Piecewise-Constant PDF sampling.

    :param key: jnp.ndarray(float32), [2,], random number generator.
    :param bins: jnp.ndarray(float32), [num_bins + 1].
    :param weights: jnp.ndarray(float32), [num_bins].
    :param num_samples: int, the number of samples.
    :param randomized: bool, use randomized samples.

    :return: z_samples: jnp.ndarray(float32), [num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = jnp.sum(weights)
    padding = jnp.maximum(0, eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = jnp.minimum(1, jnp.cumsum(pdf[:-1]))
    cdf = jnp.concatenate([
        jnp.zeros((1,)), cdf,
        jnp.ones((1,))
    ],
        axis=-1)  # [num_bins + 1]

    # Draw uniform samples.
    if randomized:
        # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = jax.random.uniform(key, (num_samples,))
    else:
        # Match the behavior of random.uniform() by spanning [0, 1-eps].
        u = jnp.linspace(0., 1. - jnp.finfo('float32').eps, num_samples)

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[None, :] >= cdf[:, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = jnp.max(jnp.where(mask, x[:, None], x[:1, None]), -2)
        x1 = jnp.min(jnp.where(~mask, x[:, None], x[-1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    # Prevent gradient from backprop-ing through `samples`.
    return jax.lax.stop_gradient(samples)


def sample_pdf(key, bins, weights, origin, direction, z_vals, num_samples, randomized):
    """Hierarchical sampling.

    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [num_bins + 1].
        weights: jnp.ndarray(float32), [num_bins].
        origin: jnp.ndarray(float32), [3], ray origin.
        direction: jnp.ndarray(float32), [3], ray direction.
        z_vals: jnp.ndarray(float32), [num_coarse_samples].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.

    Returns:
        z_vals: jnp.ndarray(float32), [num_coarse_samples + num_fine_samples,].
        points: jnp.ndarray(float32), [num_coarse_samples + num_fine_samples, 3].
    """
    z_samples = piecewise_constant_pdf(key, bins, weights, num_samples, randomized)
    # Compute united z_vals and sample points
    z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
    coords = cast_ray(z_vals, origin, direction)
    return z_vals, coords


class Renderer(eqx.Module):
    num_coarse_samples: int
    num_fine_samples: int
    near: float
    far: float
    noise_std: Optional[float]
    white_bkgd: bool
    lindisp: bool
    randomized: bool

    # make this work on a grid, must use correct ray_directions and stochasticity

    def render_nerf_pixel(self, nerf: NeRF, ray_origin, ray_direction, key: jax.Array,
                          state: Optional[eqx.nn.State] = None):
        """ 
        Render a single pixel from an NeRF based on a ray origin and direction
        :parameter nerf: the NeRF model to be rendered
        :parameter ray_origin: (3,) shaped array representing the point in space from which the ray, along which we want to render the NeRF, is cast.
        :parameter ray_direction: (3,) shaped array representing the direction of the ray
        :parameter key: jax PRNG key
        :parameter state: optional state for the NeRF model if needed (will not be updated or returned by this function)
        """
        key_gen = key_generator(key)

        # Normalize ray directions
        viewdir = ray_direction / jnp.linalg.norm(ray_direction)  # [3,]

        # Sample along rays
        z_vals, samples = sample_along_ray(  # [self.num_coarse_samples,] and [self.num_coarse_samples, 3]
            next(key_gen),
            ray_origin,  # [3,]
            ray_direction,  # [3,] 
            self.num_coarse_samples,
            self.near,
            self.far,
            self.randomized,
            self.lindisp,
        )

        # Run coarse model
        if nerf.coarse_model.is_stateful and state is not None:
            substate = state.substate(nerf.coarse_model)
            raw_rgb, raw_sigma, substate = jax.vmap(nerf.coarse_model, (0, None, None))(samples, viewdir, substate)
            # new_state = new_state.update(substate)  # in this project we don't need to update the state inside the model so we don't
        else:
            raw_rgb, raw_sigma = jax.vmap(nerf.coarse_model, (0, None))(samples, viewdir)

        # Add noise to regularize the density predictions if needed
        raw_sigma = maybe_add_gaussian_noise(
            next(key_gen),
            raw_sigma,
            self.noise_std,
            self.randomized,
        )
        rgb = jax.nn.sigmoid(raw_rgb)
        sigma = jax.nn.relu(raw_sigma)

        # Volumetric rendering
        computed_rgb, weights, depth = volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            ray_direction,
            white_bkgd=self.white_bkgd,
        )

        return_value = {
            "coarse_rgb": computed_rgb,
            "coarse_depth": depth
        }

        # hierarchical sampling based on coarse predictions
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_vals, samples = sample_pdf(
            next(key_gen),
            z_vals_mid,
            weights[..., 1:-1],
            ray_origin,
            ray_direction,
            z_vals,
            self.num_fine_samples,
            self.randomized,
        )

        # Run fine model
        if nerf.fine_model.is_stateful and state is not None:
            substate = state.substate(nerf.fine_model)
            raw_rgb, raw_sigma, substate = jax.vmap(nerf.fine_model, (0, None, None))(samples, viewdir, substate)
            # new_state = new_state.update(substate)  # in this project we don't need to update the state inside the model so we don't
        else:
            raw_rgb, raw_sigma = jax.vmap(nerf.coarse_model, (0, None))(samples, viewdir)

        raw_sigma = maybe_add_gaussian_noise(
            next(key_gen),
            raw_sigma,
            self.noise_std,
            self.randomized,
        )
        rgb = jax.nn.sigmoid(raw_rgb)
        sigma = jax.nn.relu(raw_sigma)

        computed_rgb, weights, depth = volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            ray_direction,
            white_bkgd=self.white_bkgd,
        )

        return_value["fine_rgb"] = computed_rgb
        return_value["fine_depth"] = depth
        return return_value


class ViewReconstructor(eqx.Module):
    """
    Reconstruct an image from a NeRF model.
    """
    renderer: Renderer
    num_coarse_samples: int
    num_fine_samples: int
    near: float
    far: float
    noise_std: Optional[float]
    white_bkgd: bool
    lindisp: bool
    randomized: bool
    height: int
    width: int
    folder: str
    batch_size: int
    key: jax.Array
    ray_directions: jax.Array
    ray_origins: jax.Array
    focal: float

    def __init__(self,
                 num_coarse_samples: int,
                 num_fine_samples: int,
                 near: float,
                 far: float,
                 noise_std: Optional[float],
                 white_bkgd: bool,
                 lindisp: bool,
                 randomized: bool,
                 height: int,
                 width: int,
                 folder: str,
                 key: jax.Array,
                 ):
        """
        Initialize the ImageReconstructor
        """
        self.height = height
        self.width = width
        self.focal = SyntheticScenesHelper.get_focal(folder)
        self.key = key
        self.renderer = Renderer(
            num_coarse_samples=num_coarse_samples,
            num_fine_samples=num_fine_samples,
            near=near,
            far=far,
            noise_std=noise_std,
            white_bkgd=white_bkgd,
            lindisp=lindisp,
            randomized=randomized
        )

    def __call__(self, nerf: NeRF, ray_directions, ray_origins, state: Optional[eqx.nn.State] = None):
        """
        Render the image and depth map from a NeRF model.
        :param nerf: the NeRF model to be rendered
        :param ray_directions: the directions of the rays to be cast
        :param ray_origins: the origins of the rays to be cast
        :param state: optional state for the NeRF model if needed (will not be updated or returned by this function)
        :return: rendered_image, depth_map, both as jnp.ndarray(float32), [height, width, 3] and [height, width] respectively
        """
        # Flatten rays and repeat origins for each pixel
        ray_origins_flat = jnp.tile(self.ray_origins[None, :], (self.height * self.width, 1))  # [H*W, 3]
        ray_directions_flat = self.ray_directions.reshape(-1, 3)  # [H*W, 3]

        # Generate random keys for each pixel
        keys = jax.random.split(self.key, self.height * self.width)

        # Vectorize rendering across all pixels
        results = jax.vmap(self.renderer.render_nerf_pixel, in_axes=(None, 0, 0, 0, None))(
            nerf,
            ray_origins_flat,
            ray_directions_flat,
            keys,
            state
        )

        # Extract and reshape RGB values
        rgbs = results["fine_rgb"]
        rendered_image = rgbs.reshape((self.height, self.width, 3))
        depth = results["fine_depth"].reshape((self.height, self.width))
        return rendered_image, depth



class SyntheticScenesHelper:
    # just a collection of methods related to each other
    @staticmethod
    def get_focal(folder: str) -> float:
        # based on https://github.com/vsitzmann/deepvoxels?tab=readme-ov-file#coordinate-and-camera-parameter-conventions
        with open(f"{folder}/intrinsics.txt", 'r') as intrinsics_file:
            first_line = next(intrinsics_file)
            focal_length = float(first_line.split(' ')[0])
        return focal_length

    @staticmethod
    def generate_rays(height: int, width: int, focal: float, pose: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:  # TODO: use this to get ray directions, this information is in pose/base_name.txt
        # focal length comes from intrinsic_file
        """
        Generate rays for volumetric rendering
        :param height: height of the image
        :param width: width of the image
        :param focal: focal length
        :param pose: camera pose
        :return: ray origin and ray direction
        """

        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

        transformed_i = (i - width * 0.5) / focal
        transformed_j = (j - height * 0.5) / focal
        k = -np.ones_like(i)  # z-axis

        directions = np.stack([transformed_i, transformed_j, k], axis=-1)  # 2-d grid of 3-d vectors
        camera_matrix = pose[:3, :3]
        ray_directions = np.einsum("ijl,kl->ijk", directions,
                                   camera_matrix)  # multiply each vector in the grid by camera_matrix
        ray_origin = pose[:3, -1]

        return ray_origin, ray_directions

    @classmethod
    def create_numpy_arrays(cls, folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        file_base_names = [
            file_name.removesuffix('.png')
            for file_name in os.listdir(f"{folder}/rgb")
        ]
        file_base_names = sorted(file_base_names)

        focal_length = cls.get_focal(folder)

        # first, we determine the sizes of arrays that we need
        num_images = len(file_base_names)

        with Image.open(f"{folder}/rgb/{file_base_names[0]}.png") as example_image:
            image_size = example_image.size

        images_shape = (num_images,) + image_size + (3,)
        images = np.empty(images_shape, dtype=np.float32)

        poses_shape = (num_images, 4, 4)
        poses = np.empty(poses_shape, dtype=np.float32)

        origins_shape = (num_images, 3)
        directions_shape = (num_images,) + image_size + (3,)

        ray_origins = np.empty(origins_shape, dtype=np.float32)
        ray_directions = np.empty(directions_shape, np.float32)

        # next, we fill those arrays
        for index, base_name in enumerate(file_base_names):
            image_path = f"{folder}/rgb/{base_name}.png"
            pose_path = f"{folder}/pose/{base_name}.txt"

            with Image.open(image_path) as image:
                images[index] = np.asarray(image, dtype=np.float32) / 255.

            # following is based on https://github.com/vsitzmann/deepvoxels/blob/a12171a77b68980b9b025c70c117e92af973f20b/data_util.py#L53
            flat_pose = np.loadtxt(pose_path, dtype=np.float32)
            pose = flat_pose.reshape((4, 4))

            poses[index] = pose

            origin, directions = cls.generate_rays(
                height=image_size[0],
                width=image_size[1],
                focal=focal_length,
                pose=pose
            )
            ray_origins[index] = origin
            ray_directions[index] = directions

        return images, poses, ray_origins, ray_directions


class SyntheticScenesDataLoader:
    images: jax.Array
    poses: jax.Array
    ray_origins: jax.Array
    ray_directions: jax.Array
    batch_size: int
    poses_per_batch: int
    _pixels_per_pose: int

    def __iter__(self):
        """ 
        :yields: three arrays:
            ray origins
            ray directions
            prng_key
            ground truth
        """
        with jax.default_device(self._cpu):
            key_gen = key_generator(self.initial_key)

        while True:
            with jax.default_device(self._cpu):
                key = next(key_gen)

            ray_origins, ray_directions, return_key, ground_truth_pixel_values = self._prepare_arrays(key)

            yield (
                jax.device_put(ray_origins, self._gpu),
                jax.device_put(ray_directions, self._gpu),
                jax.device_put(return_key, self._gpu),
                jax.device_put(ground_truth_pixel_values, self._gpu)
            )

    @(lambda f: jax.jit(f, device=jax.devices('cpu')[0], static_argnums=0))
    def _prepare_arrays(self, key: jax.Array):
        with jax.default_device(self._cpu):
            poses_key, directions_key, return_key = jax.random.split(key, 3)

            height, width = self.images.shape[1:3]
            grid_size = height * width
            dataset_size = self.images.shape[0]

            selected_camera_poses_idx = jax.random.choice(poses_key, dataset_size, shape=(self.poses_per_batch,))
            selected_directions_idx_flat = jax.random.choice(directions_key, grid_size, shape=(
                self.poses_per_batch, self._pixels_per_pose))  # flat (raveled) indices into grid
            selected_camera_poses_idx = jnp.broadcast_to(selected_camera_poses_idx[:, None],
                                                         (self.poses_per_batch, self._pixels_per_pose))

            selected_directions_idx_grid = jax.vmap(
                lambda flat_index: jnp.unravel_index(flat_index, shape=(height, width))
            )(selected_directions_idx_flat)  # multi indices into (height, width) grid

            selected_directions_idx = (
                                          selected_camera_poses_idx,) + selected_directions_idx_grid  # multi indices into (dataset_size, height, width) array

            ray_origins = self.ray_origins[selected_camera_poses_idx]
            ray_directions = self.ray_directions[selected_directions_idx]
            ground_truth_pixel_values = self.images[selected_directions_idx]

            ray_origins = ray_origins.reshape((self.batch_size, 3))
            ray_directions = ray_directions.reshape((self.batch_size, 3))
            ground_truth_pixel_values = ground_truth_pixel_values.reshape((self.batch_size, 3))

            return ray_origins, ray_directions, return_key, ground_truth_pixel_values

    def __init__(self, split: str, name: str, batch_size: int, poses_per_batch: int,
                 base_path: str = "example_data/synthetic_scenes", size_limit: int = -1, *, key: jax.Array):
        self._cpu = jax.devices('cpu')[0]
        self._gpu = jax.devices('gpu')[0]


        folder = f"{base_path}/{split}/{name}"
        if not os.path.exists(folder):
            raise ValueError(f"Following folder does not exist: {folder}")
        if batch_size % poses_per_batch:
            raise ValueError(
                f"batch_size should be divisible by poses_per_batch. Got {batch_size=} but {poses_per_batch=}  - note that {batch_size % poses_per_batch=}.")

        self.batch_size = batch_size
        self.poses_per_batch = poses_per_batch
        self._pixels_per_pose = batch_size // poses_per_batch

        with jax.default_device(self._cpu):
            # if we've already stored everything in a single big npz file, just load that
            target_path = f"{folder}/pre_processed.npz"
            if os.path.exists(target_path):
                pre_processed = np.load(target_path)
                self.images = jnp.asarray(pre_processed['images'][:size_limit])
                self.poses = jnp.asarray(pre_processed['poses'][:size_limit])
                self.ray_origins = jnp.asarray(pre_processed['ray_origins'][:size_limit])
                self.ray_directions = jnp.asarray(pre_processed['ray_directions'][:size_limit])
            else:  # otherwise, create said npz file
                print(f"creating npz archive for {split}, {name}.")
                images, poses, ray_origins, ray_directions = SyntheticScenesHelper.create_numpy_arrays(folder)
                self.images = jnp.asarray(images[:size_limit])
                self.poses = jnp.asarray(poses[:size_limit])
                self.ray_origins = jnp.asarray(ray_origins[:size_limit])
                self.ray_directions = jnp.asarray(ray_directions[:size_limit])
                np.savez(target_path, images=images, poses=poses, ray_origins=ray_origins,
                         ray_directions=ray_directions)
                print(f"    finished creating {target_path}")

        self.initial_key = jax.device_put(key, self._cpu)

# # the following was just a bad idea
# class _SharedArray:
#     """ 
#     This is a descriptor (see https://docs.python.org/3/howto/descriptor.html) that can return views into a shared array
#     Here shared means that it can be shared between processes and workers
#     """
#     def __init__(self, c_type:str):
#         self.c_type = c_type

#     def __set_name__(self, owner, name):
#         if not hasattr(owner, "_shared_arrays"):
#             owner._shared_arrays = dict()
#         self.name = name
#         owner._shared_arrays[name] = dict()

#     def __set__(self, obj, value:np.ndarray):
#         data_dict = obj._shared_arrays[self.name]
#         data_dict["shape"] = value.shape
#         data_dict["shared_array_base"] = multiprocessing.Array(self.c_type, value.flatten(), lock=False)  # lock=False because we assume we'll only wan to read from the array
#         data_dict["np_dtype"] = value.dtype

#     def __get__(self, obj, objtyp=None):
#         data_dict = obj._shared_arrays[self.name]
#         return np.frombuffer(data_dict["shared_array_base"], dtype=data_dict["np_dtype"]).reshape(data_dict["shape"])  # returns a view into the shared array


# class _SyntheticScenesContainer:
#     """ 
#     Creates singletons that contain views into shared arrays.
#     """
#     containers: ClassVar[dict] = {}  # class variable because we want singletons
#     images = _SharedArray('f')
#     poses = _SharedArray('f')
#     ray_origins = _SharedArray('f')
#     ray_directions = _SharedArray('f')
#     _shared_arrays: dict  # under the hood, this holds the actual data

#     def __new__(cls, split:str, name: str, base_path: str):
#         """ 
#         If a container for (split, name, base_path) was previously created, just return that one, otherwise, create one and store it.
#         """
#         key = (split, name, base_path)
#         # don't use cls.containers.setdefault because that still has a new instance created (and then garbage collected) every time you call with the same split, name, base_path
#         if key not in cls.containers:
#             cls.containers[key] = super().__new__(cls)
#         return cls.containers[key]

#     def __init__(self, split:str, name:str, base_path:str):
#         folder = f"{base_path}/{split}/{name}"
#         if not os.path.exists(folder):
#             raise ValueError(f"Following folder does not exist: {folder}")

#         # if we've already stored everything in a single big npz file, just load that
#         target_path = f"{folder}/pre_processed.npz"
#         if os.path.exists(target_path):
#             pre_processed = np.load(target_path)
#             self.images = pre_processed['images']
#             self.poses = pre_processed['poses']
#             self.ray_origins = pre_processed['ray_origins']
#             self.ray_directions = pre_processed['ray_directions']
#         else: # otherwise, create said npz file
#             print(f"creating npz archive for {split}, {name}.")
#             images, poses, ray_origins, ray_directions = SyntheticScenesHelper.create_numpy_arrays(folder)
#             self.images = images
#             self.poses = poses
#             self.ray_origins = ray_origins
#             self.ray_directions = ray_directions
#             np.savez(target_path, images=images, poses=poses, ray_origins=ray_origins, ray_directions=ray_directions)
#             print(f"    finished creating {target_path}")

#         self._len = np.prod(self.images.shape[:-1])

#     def __len__(self):
#         return self._len


# class SyntheticScenesDataset(IterableDataset):

#     def __init__(self, split:str, name:str, base_path:str="./synthetic_scenes", *, initial_key:jax.Array):
#         self.data = _SyntheticScenesContainer(split, name, base_path)  # singleton that can return views into shared arrays
#         cpu = jax.devices('cpu')[0]
#         self.initial_key = jax.device_put(initial_key, cpu)

#     def __iter__(self):
#         cpu = jax.devices('cpu')[0]
#         with jax.default_device(cpu):
#             worker_info = get_worker_info()  # fom torch.utils.data
#             if not worker_info:
#                 return self.iterate_from_jax_key(self.initial_key)  # we don't have to split the key over workers
#             worker_keys = jax.random.split(self.initial_key, num=worker_info.num_workers)
#             key = worker_keys[worker_info.id]
#             return self.iterate_from_jax_key(key)


#     def iterate_from_jax_key(self, key:jax.Array):
#         #print(f"Starting SyntheticScenesDataset iterable from key={jax.random.key_data(key)}.")
#         cpu = jax.devices('cpu')[0]
#         height, width = self.data.images.shape[1:3]
#         grid_size = height * width
#         dataset_size = self.data.images.shape[0]
#         with jax.default_device(cpu):
#             key = jax.device_put(key)
#             key_gen = key_generator(key)

#             while True:
#                 key = next(key_gen)
#                 poses_key, directions_key, return_key = jax.random.split(key, 3)

#                 selected_camera_pose_id = jax.random.choice(poses_key, dataset_size)
#                 selected_direction_id_flat = jax.random.choice(directions_key, grid_size)  # flat (raveled) index into grid
#                 # make numpy arrays out of the jax arrays
#                 selected_camera_pose_id = np.asarray(selected_camera_pose_id)
#                 selected_direction_id_flat = np.asarray(selected_direction_id_flat)

#                 selected_direction_id_grid = np.unravel_index(selected_direction_id_flat, shape=(height, width))

#                 selected_direction_id = (selected_camera_pose_id,)+ selected_direction_id_grid #  multi index into (dataset_size, height, width) array

#                 ray_origin = self.data.ray_origins[selected_camera_pose_id]
#                 ray_direction = self.data.ray_directions[selected_direction_id]
#                 ground_truth_pixel_value = self.data.images[selected_direction_id]

#                 yield (  # make jax arrays out of the numpy ones before we yield them
#                     jnp.asarray(ray_origin), 
#                     jnp.asarray(ray_direction), 
#                     return_key, 
#                     jnp.asarray(ground_truth_pixel_value)
#                     )
