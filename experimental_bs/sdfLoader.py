from typing import Tuple, Union, List, Callable

import pymeshlab
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import jax


def ply_to_xyz(file_name: str) -> None:
    fp_ply = Path.cwd().parent.joinpath("example_data", f"{file_name}.ply")
    fp_xyz = Path.cwd().parent.joinpath("example_data", f"{file_name}.xyz")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(fp_ply))
    ms.save_current_mesh(
        str(fp_xyz),
        save_vertex_normal=True,
    )
    return None


def load_xyz(file_name: str, keep_aspect_ratio: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from .xyz file and normalize it to [-1, 1]^3, return coords and normals
    :param file_name: name of the file without extension
    :param keep_aspect_ratio: if True, keep the aspect ratio of the point cloud
    :return: coords and normals of the point cloud
    """
    if not Path.cwd().parent.joinpath("example_data", f"{file_name}.xyz").exists():
        if not Path.cwd().parent.joinpath("example_data", f"{file_name}.ply").exists():
            raise FileNotFoundError(f"File {file_name} does not exist")
        else:
            ply_to_xyz(file_name)

    fp_xyz = Path.cwd().parent.joinpath("example_data", f"{file_name}.xyz")
    print("Loading point cloud from", fp_xyz)
    point_cloud = np.genfromtxt(fp_xyz)
    print("loaded point cloud with shape", point_cloud.shape)
    coords = point_cloud[:, :3]
    normals = point_cloud[:, 3:]

    coords -= np.mean(coords, axis=0, keepdims=True)
    if keep_aspect_ratio:
        coord_max = np.amax(coords)
        coords_min = np.amin(coords)
    else:
        coord_max = np.amax(coords, axis=0, keepdims=True)
        coords_min = np.amin(coords, axis=0, keepdims=True)
    coords = (coords - coords_min) / (coord_max - coords_min)
    coords -= 0.5
    coords *= 2
    return coords, normals


def sdf_dataloader(batch_size, filename, key, keep_aspect_ration) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
    """
    Loads the normalized point cloud from the file and generates batches of surface and off-surface points
    :param batch_size: size of the batch
    :param filename: name of the file without extension
    :param key: random key
    :param keep_aspect_ration: if True, keep the aspect ratio of the point cloud
    :return: coords_batch, normals_batch, sdf or None if the point cloud is exhausted
    """

    surface_points = batch_size // 2

    coords, normals = load_xyz(filename, keep_aspect_ration)
    point_cloud_size = coords.shape[0]
    iterations = point_cloud_size // surface_points

    off_surface_samples = batch_size - surface_points

    # shuffle the coords and normals
    key, subkey = jax.random.split(key)
    shuffle_indices = jax.random.permutation(subkey, point_cloud_size)
    coords = coords[shuffle_indices]
    normals = normals[shuffle_indices]

    sp_sdf = jnp.zeros((surface_points, 1))
    nsp_sdf = jnp.ones((off_surface_samples, 1)) * -1

    sdf = jnp.concatenate([sp_sdf, nsp_sdf], axis=0)

    for i in range(iterations):
        key, subkey = jax.random.split(key)
        surface_coords = coords[i * surface_points:(i + 1) * surface_points]
        surface_normals = normals[i * surface_points:(i + 1) * surface_points]
        off_surface_coords = jax.random.uniform(subkey, (off_surface_samples, 3), minval=-1, maxval=1)
        off_surface_normals = jnp.ones((off_surface_samples, 3)) * -1

        coords_batch = jnp.concatenate([surface_coords, off_surface_coords], axis=0)
        normals_batch = jnp.concatenate([surface_normals, off_surface_normals], axis=0)
        yield coords_batch, normals_batch, sdf

    if point_cloud_size % surface_points != 0:
        left_over = point_cloud_size % surface_points
        compensating_for_batch_size = batch_size - left_over
        sp_sdf = jnp.zeros((left_over, 1))
        nsp_sdf = jnp.ones((compensating_for_batch_size, 1)) * -1
        sdf = jnp.concatenate([sp_sdf, nsp_sdf], axis=0)

        key, subkey = jax.random.split(key)
        surface_coords = coords[iterations * surface_points:]
        surface_normals = normals[iterations * surface_points:]
        off_surface_coords = jax.random.uniform(subkey, (compensating_for_batch_size, 3), minval=-1, maxval=1)
        off_surface_normals = jnp.ones((compensating_for_batch_size, 3)) * -1

        coords_batch = jnp.concatenate([surface_coords, off_surface_coords], axis=0)
        normals_batch = jnp.concatenate([surface_normals, off_surface_normals], axis=0)
        return coords_batch, normals_batch, sdf

    return None, None, None


def gradient(f, x):
    return jax.grad(f)(x)


def cosine_similarity(a, b):
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def sdf_loss(model, activatoin, cs, ground_truths, y_pred):
    """
    :param model: model
    :param cs: coordinates that the model is evaluated at this batch
    :param ground_truths: normal and sdf ground truths
    :param y_pred: normal and sdf predictions

    """
    gt_normals, gt_sdf = ground_truths
    sdf_pred = y_pred

    f = partial(forward, model, activation=activatoin)
    # grad_sdf = gradient(f, cs)
    grad_sdf = jax.vmap(gradient, in_axes=(None, 0))(f, cs)
    sdf_constraint = jnp.where(gt_sdf != -1, sdf_pred, jnp.zeros_like(sdf_pred))
    inter_constraint = jnp.where(gt_sdf != -1, jnp.zeros_like(sdf_pred), jnp.exp(-1e2 * jnp.abs(sdf_pred)))
    print(f"{gt_sdf.shape}, {grad_sdf.shape=}, {gt_normals.shape=}")
    cosine_sim = jax.vmap(cosine_similarity)(grad_sdf, gt_normals)


    gt_sdf = jnp.expand_dims(gt_sdf, axis=1)
    normal_constraint = jnp.where(gt_sdf != -1, 1 - cosine_sim, jnp.zeros_like(grad_sdf[:, 0]))

    grad_constraint = jnp.abs(jnp.linalg.norm(grad_sdf) - 1)

    loss_dct = {
        "sdf": jnp.abs(jnp.mean(sdf_constraint)) * 3e3,
        "inter": jnp.mean(inter_constraint) * 1e2,
        "normal": jnp.mean(normal_constraint) * 1e2,
        "grad": jnp.mean(grad_constraint) * 5e1
    }
    return loss_dct


def make_model(in_dim, hidden_dim, hidden_layers, out_dim, key):
    weights = []
    biases = []

    dims = [in_dim] + [hidden_dim] * hidden_layers + [out_dim]

    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        weights.append(jax.random.normal(subkey, (dims[i + 1], dims[i])) / jnp.sqrt(dims[i + 1]))
        biases.append(jax.random.normal(subkey, (dims[i + 1],)) / jnp.sqrt(dims[i]))

    return weights, biases


def forward(model: Tuple[List[jax.Array], List[jax.Array]], x: jax.Array, activation: Callable) -> jax.Array:
    weights, biases = model
    for w, b in zip(weights, biases):
        x = w@x + b
        x = activation(x)
    return x[0]

from functools import partial

if __name__ == "__main__":
    sdf_loader = sdf_dataloader(100, "xyzrgb_statuette", jax.random.PRNGKey(0), True)
    for coords, normals, sdf in sdf_loader:
        break
    print(f"{coords.shape=}, {normals.shape=}, {sdf.shape=}")
    # coords = jnp.array([1.1, 2.1, 3.1])
    # normals = jnp.array([1.1, 2.1, 3.1])
    model = make_model(3, 4, 3, 1, jax.random.PRNGKey(0))
    activation = jnp.sin
    f = partial(forward, model, activation=activation)

    y_pred = jax.vmap(f)(coords)
    print(y_pred.shape)

    # grads_sdf = jax.vmap(gradient, in_axes=(None, 0))(f, coords)
    # print(grads_sdf)
    ground_truths = (normals, sdf)
    loss = sdf_loss(model, activation, coords, ground_truths, y_pred)
    print(f"{loss=}")