# get nerf data
# !wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
import jax
import jax.numpy as jnp
from typing import Tuple, Union, Generator, Dict


def _generate_rays(height, width, focal, pose) -> jnp.ndarray:
    """
    Generate rays for volumetric rendering
    :param height: height of the image
    :param width: width of the image
    :param focal: focal length
    :param pose: camera pose
    :return: ray origin and ray direction
    """

    i, j = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing='xy')

    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal
    k = -jnp.ones_like(i)  # z-axis

    directions = jnp.stack([transformed_i, transformed_j, k], axis=-1)
    camera_matrix = pose[:3, :3]
    ray_direction = jnp.einsum(("ijl,kl", directions, camera_matrix))
    ray_origin = jnp.broadcast_to(pose[:3, -1], ray_direction.shape)

    return jnp.stack([ray_origin, ray_direction], axis=-1)


def _3d_points(ray_origin, ray_direction, config: Dict, key_gen: Union[Generator, None] = None) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
    """
    Compute 3D points along the rays for volumetric rendering
    :param ray_origin: origin of the rays
    :param ray_direction: direction of the rays
    :param config: configuration of the rendering
    :param key_gen: key generator
    :return: 3D points and z-values
    """

    z_vals = jnp.linspace(0., 1., config["num_samples"])
    if key_gen is not None:
        bd_diff = config["far"] - config["near"]
        z_vals += jax.random.uniform(next(key_gen), z_vals.shape) * bd_diff / config["num_samples"]
    # r(t) = o + t * d
    points = ray_origin[Ellipsis, None, :] + ray_direction[Ellipsis, None, :] * z_vals[Ellipsis, :, None]
    return points, z_vals


def _radiance_field(model_fn, points, config: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the radiance field for the given points
    :param model_fn: model function
    :param points: 3D points
    :param config: configuration of the rendering
    :return: RGB values and opacity
    """
    model_out = jax.lax.map(model_fn, points.reshape((-1, config["batch_size"], 3)))
    radiance_field = model_out.reshape(points.shape[:-1] + (4,))

    opacity = jax.nn.relu(radiance_field[Ellipsis, 3])
    rgb = jax.nn.sigmoid(radiance_field[Ellipsis, :3])
    return rgb, opacity


def _adjacent_distances(z_vals, ray_direction, config: Dict) -> jnp.ndarray:
    """
    Compute the distances between adjacent z-values
    :param z_vals: z-values
    :param ray_direction: direction of the rays
    :param config: configuration of the rendering
    :return: distances between adjacent z-values
    """
    dists = z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1]
    dists = jnp.concatenate(
        [
            dists, jnp.broadcast_to(
            [config["epsilon"]],
            dists[Ellipsis, :1].shape)
        ],
        axis=-1
    )
    dists = dists * jnp.linalg.norm(ray_direction[Ellipsis, None, :], axis=-1)
    return dists


def _weights(opacity, dists, config: Dict) -> jnp.ndarray:
    """
    Compute the weights for the radiance field
    :param opacity: opacity
    :param dists: distances between adjacent z-values
    :param config: configuration of the rendering
    :return: weights for the radiance field
    """
    density = jnp.exp(-opacity * dists)
    alpha = 1. - density
    clipped_diff = jnp.clip(1.0 - alpha, config["epsilon"], 1.0)

    transmittance = jnp.cumprod(
        jnp.concatenate([
            jnp.ones_like(clipped_diff[Ellipsis, :1]),
            clipped_diff[Ellipsis, :-1]],
            axis=-1),
        axis=-1
    )
    weights = alpha * transmittance
    return weights


def _volume_render(model_fn, rays, config: Dict, key_gen=None) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Volume rendering
    :param model_fn: model function
    :param rays: rays
    :param config: configuration of the rendering
    :param key_gen: key generator
    :return: RGB map, depth map, disparity map, opacity
    """
    ray_origin, ray_direction = rays

    points, z_vals = _3d_points(ray_origin, ray_direction, key_gen)

    rgb, opacity = _radiance_field(model_fn, points, config)

    dists = _adjacent_distances(z_vals, ray_direction, config)

    weights = _weights(opacity, dists, config)

    rgb_map = jnp.sum(weights[Ellipsis, None] * rgb, axis=-2)
    depth_map = jnp.sum(weights * z_vals, axis=-1)
    scaled_depth_map = depth_map / jnp.sum(weights, axis=-1)

    disparity_map = 1. / jnp.max(config["epsilon"], scaled_depth_map)

    return rgb_map, depth_map, disparity_map, opacity


def ray_to_ndc(origins, directions, focal, w, h, near=1.) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert rays to normalized device coordinates
    :param origins: ray origins
    :param directions: ray directions
    :param focal: focal length
    :param w: width of the image
    :param h: height of the image
    :param near: near plane
    :return: ray origins and ray directions in normalized device coordinates
    """
    # Shift ray origins to near plane
    t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = tuple(jnp.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(jnp.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = jnp.stack([o0, o1, o2], -1)
    directions = jnp.stack([d0, d1, d2], -1)
    return origins, directions
