""" 
This module implements an example training loop for training INRs (not with forward maps though).
If you prefer to have your training loop be part of some Trainer class, look at hypernetwork_utils.training for inspiration.
"""
from typing import Callable, Union

import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import optax
import equinox as eqx

from common_jax_utils import key_generator

from inr_utils.sampling import Sampler
from inr_utils.callbacks import Callback

def make_inr_train_step_function(target_function:Callable, loss_function:Callable, sampler:Sampler, optimizer:optax.GradientTransformation):
    """make_inr_train_step_function
    Make an INR train_step function to learn the target_function using the given loss_function, sampler, and optimizer

    :param target_function: function to be approximated by the INR
    :param loss_function: function to compute the loss based on pred_val and true_val
        should have the following signature
        :param pred_val: jax.Array of predicted values
        :param true_val: jax.Array of target values with same shape as pred_val
        :return: scalar jax.Array containing the loss value
    :param sampler: sampler for coordinates at which to evaluate the inr and target_function
        should have the following signature:
        :param key: jax.Array functioning as a prng key
        :return: jax.Array of locations at which to probe the inr and target function
    :param optimizer: optax optimizer

    :return: train_step function, evaluate_loss function
        train_step has the follwing signature:
            :param inr: inr model to be optimized
            :param optimizer_state: state of the used (optax) optimizer
            :param key: key for prng
            :return: updated inr, updated optimizer_state, loss for this step
        evaluate_loss has the following signature
            :param inr: eqx.Module to be evaluated at locations
            :param locations: (batch_size, n_dim) shaped jax.Array of locations at which to evaluate the inr and target_function
            :return: loss value (should be a scalar, assuming loss_function has the correct signature)

    NB the train_step function is decorated with eqx.filter_jit, so it can be jit-ed at the highest level.
    This means that ideally, the loss_function, target_function, and sampler should not be jit-ed.
    But they should all be jit-able! 
    """
    def evaluate_loss(inr: eqx.Module, locations:jax.Array):
        """
        Evaluate the loss of the inr at the locations
        :param inr: eqx.Module to be evaluated at locations
        :param locations: (batch_size, n_dim) shaped jax.Array of locations at which to evaluate the inr and target_function
        :return: loss value (should be a scalar, assuming loss_function has the correct signature)
        """
        pred_val = jax.vmap(inr)(locations)
        true_val = jax.vmap(target_function)(locations)
        return loss_function(pred_val, true_val)

    @eqx.filter_jit  # this is likely the highest level at which we can reasonably jit
    def train_step(inr:eqx.Module, optimizer_state:optax.OptState, key:jax.Array):
        """take one train step with the inr model and optimizer

        :param inr: inr model to be optimized
        :param optimizer_state: state of the used (optax) optimizer
        :param key: key for prng
        :return: updated inr, updated optimizer_state, loss for this step 

        NB this function is decorated with eqx.filter_jit
        """
        locations = sampler(key)
        loss, grad = eqx.filter_value_and_grad(evaluate_loss)(inr, locations)
        grad = jax.tree.map(
            lambda x: jnp.conjugate(x) if jnp.iscomplexobj(x) else x,
            grad
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state, inr)
        inr = eqx.apply_updates(inr, updates)
        return inr, optimizer_state, loss
    return train_step, evaluate_loss

def train_inr(
        inr:eqx.Module,
        target_function: Union[Callable, eqx.Module],
        loss_function: Callable,
        sampler:Sampler, 
        optimizer:optax.GradientTransformation,
        steps: int,
        key: jax.Array,
        use_wandb: bool,
        after_step_callback: Union[None, Callable, Callback] = None,
        after_training_callback: Union[None, Callable] = None,
        optimizer_state: Union[None, optax.OptState] = None
        ):
    """train_inr train an inr

    :param inr: inr to be trained
    :param target_function: target_function to be learned
    :param loss_function: loss_function to evaluate quality of inr
        should have the folllowing signature
        :param pred_val: jax.Array of predicted values
        :param true_val: jax.Array of target values with same shape as pred_val
        :return: scalar jax.Array containing the loss value
    :param sampler: sampler for coordinates at which to evaluate the inr and target_function
        should have the following signature:
        :param key: jax.Array functioning as a prng key
        :return: jax.Array of locations at which to probe the inr and target function
    :param optimizer: optax optimizer
    :param steps: number of training steps to take
    :param key: key for PRNG
    :param after_step_callback: optional function or Callback to be called after each step, 
        defaults to None
        Should have the following signature:
        :param step: integer indicating what step of the training was just performed
        :param loss: loss value after last training step
        :param inr: inr after last training step
        :param optimizer_state: state of the optimizer
        :return: None (output is ignored)
        This can e.g. be used to plot the inr every so often
    :param after_training_callback: optional function or Callback to be called after training
        if None is provided, the losses will be plotted instead.
        if a Callable is provided, it will be passed the following key-word arguments:
            :param losses: list of loss values during training
            :param inr: final inr after training
            :param target_function: target function that was being approximated
            :param sampler: sampler used for probing target_function and inr during training
            :param evaluate_loss: function for evaluating the loss of the inr at given locations
                has the following signature:
                    :param inr: the model being probed
                    :param locations: the locations at which to probe the model and target_function
                    :return: loss value
            :param loss_function: loss_function used during training
            :param key: key for PRNG
        its output will be returned as additional_output
    :param optimizer_state: optional optax.OptState in case you want to continue a training process instead of starting a new one
        If None is provided, the state is initialized using optimizer.init
    :return: inr, losses, optimizer_state, evaluate_loss, additional_output

    NB the train_step function is decorated with eqx.filter_jit, so it can be jit-ed at the highest level.
    This means that ideally, the loss_function, target_function, and sampler should not be jit-ed.
    But they should all be jit-able! 
    """
    train_step, evaluate_loss = make_inr_train_step_function(
        target_function=target_function,
        loss_function=loss_function,
        sampler=sampler,
        optimizer=optimizer
    )
    key_gen = key_generator(key)  # upon next, this just splits the key it has internally stored, and returns the second half of that split
    if optimizer_state is None:
        optimizer_state = optimizer.init(eqx.filter(inr, eqx.is_array))

    losses = steps*[None]

    for step in range(steps):  # the actual training loop
        inr, optimizer_state, loss = train_step(inr, optimizer_state, next(key_gen))
        losses[step] = loss
        after_step_callback(step, loss, inr, optimizer_state)

    if after_training_callback is None:
        plt.plot(losses)
        plt.show()
        additional_output = None
    else:
        additional_output = after_training_callback(
            losses=losses, 
            inr=inr, 
            target_function=target_function, 
            sampler=sampler, 
            evaluate_loss=evaluate_loss, 
            loss_function=loss_function,
            key=next(key_gen)
            )
    if use_wandb:  # upload the model to weights and biases
        import wandb
        import tempfile
        # first save the model to a temporary file
        with tempfile.NamedTemporaryFile() as f:
            eqx.tree_serialise_leaves(f, inr)
            wandb.log_model(path=f.name, name='inr.eqx')
    return inr, losses, optimizer_state, evaluate_loss, additional_output

