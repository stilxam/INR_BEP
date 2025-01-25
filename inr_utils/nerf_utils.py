"""
Based on the jaxnerf implementation from google-research.
"""
from typing import Optional

import jax
from jax import numpy as jnp
import equinox as eqx

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


def volumetric_rendering(rgb:jax.Array, sigma:jax.Array, z_vals:jax.Array, direction:jax.Array, white_bkgd:bool):  # TODO include reference to original code
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
        jnp.cumprod(1.0 - alpha + eps, axis=-1)  # TODO find out if it may be more stable to just take the exponential of the cummulative sum
    ],
        axis=-1)
    alpha = jnp.concatenate([alpha, jnp.ones_like(alpha[:1], alpha.dtype)])  # [num_samples] essentially, treat the far end of the rendering box as opaque because the final distance should be 'infinity'
    weights =  alpha * accummulated_transmittance  # [num_samples]

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

    def render_nerf_pixel(self, nerf:NeRF, ray_origin, ray_direction, key:jax.Array, state:Optional[eqx.nn.State]=None):
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

