""" 
Some common loss functions for training INRs.
"""
import jax
from jax import numpy as jnp

def mse_loss(pred_val: jax.Array, true_val: jax.Array):
    return jnp.mean(jnp.sum(jnp.square(pred_val - true_val), axis=-1))

def scaled_mse_loss(pred_val:jax.Array, true_val:jax.Array, eps=1e-6):  
    """ 
    A scaled version of the mse loss, where the loss on each batch gets scaled 
    so that 1 means you correctly predict the mean of the true values over that batch

    :parameter pred_val: the predicted values
    :parameter true_val: the true values
    :parameter eps: a small value to avoid division by zero
    """
    mse = mse_loss(pred_val, true_val)
    mean_true_val = true_val.mean(axis=0, keepdims=True)
    scaling = mse_loss(mean_true_val, true_val)
    return mse/(scaling + eps)


def cosine_similarity(a:jax.Array, b:jax.Array)->jax.Array:
    """
    Compute the cosine similarity between arrays a and b
    """
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def sdf_loss(gt_normals, y, y_pred, y_grad) -> jax.Array:
    """
    Compute the loss for the sdf values, the intersection values, the normal values, and the gradient values
    as per the SIREN paper and code (https://github.com/vsitzmann/siren/blob/master/loss_functions.py)

    if the model uses relu, y_grad = 0 can break the loss function and make gradients loss nan
    :param gt_normals: ground truth normals
    :param y: ground truth sdf values
    :param y_pred: predicted sdf values
    :param y_grad: gradients of the sdf values
    :return: loss_array, containing the loss values for sdf, inter, normal, and grad
    """

    sdf_constraint = jnp.where(y != -1, y_pred, jnp.zeros_like(y_pred))
    inter_constraint = jnp.where(y != -1, jnp.zeros_like(y_pred), jnp.exp(-1e2 * jnp.abs(y_pred)))
    cosine_sim = jax.vmap(cosine_similarity)(y_grad, gt_normals)
    if len(cosine_sim.shape) == 1:
        y = jnp.expand_dims(y, axis=1)

    normal_constraint = jnp.where(y != -1, 1 - cosine_sim, jnp.zeros_like(y_grad[:, 0]))
    gradient_constraint = jnp.abs(jnp.linalg.norm(y_grad) - 1)
    # loss_dct = {
    #     "sdf": jnp.abs(jnp.mean(sdf_constraint)) * 3e3,
    #     "inter": jnp.mean(inter_constraint) * 1e2,
    #     "normal": jnp.mean(normal_constraint) * 1e2,
    #     "grad": jnp.mean(gradient_constraint) * 5e1
    # }
    loss_array = jnp.array([
        jnp.abs(jnp.mean(sdf_constraint)) * 3e3,
        jnp.mean(inter_constraint) * 1e2,
        jnp.mean(normal_constraint) * 1e2,
        jnp.mean(gradient_constraint) * 5e1
    ])
    return loss_array

