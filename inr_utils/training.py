""" 
This module implements an example training loop for training INRs (not with forward maps though).
If you prefer to have your training loop be part of some Trainer class, look at hypernetwork_utils.training for inspiration.
"""
from typing import Callable, Union, Optional

import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import optax
import equinox as eqx
from jaxtyping import PyTree

import numpy as np

from common_jax_utils import key_generator
from common_jax_utils.types import register_type

from inr_utils.sampling import Sampler
from inr_utils.callbacks import Callback
from inr_utils.losses import LossEvaluator


# bunch of dummy classes for type annotations
@register_type
class Schedule:
    pass

@register_type
class DataLoader:
    pass


# collection of code for creating more elaborate optimizers
class OptimizerFactory:
    """ 
    This is a collection of functions for creating optimizers
    Because much of optax works with factory functions, these don't always
    play nice with the config_realization tools from common_dl_utils when it comes to type annotations
    (e.g. the type annotation for mask in adamw is Optional[Union[Any, Callable[[optax.Params], Any]]])
    """
    @staticmethod
    def single_optimizer(
            optimizer_type: type[optax.GradientTransformation],
            optimizer_config: dict,
            optimizer_mask:Optional[Callable]=None,
            learning_rate_schedule:Union[None, Schedule, list[Schedule], tuple[Schedule, ...]]=None,
            schedule_boundaries:Optional[list[int]]=None,
            )->optax.GradientTransformation:
        """single_optimizer
        Create a single optimizer (without synced parameters) optionally with a learning rate schedule and a mask function

        :param optimizer_type: factory for an optax.GradientTransformation (e.g. optax.adam)
        :param optimizer_config: configuration for the optimizer
        :param optimizer_mask: optional callable for masking parameters, defaults to None
            if this is not None, it is added to (a copy of) optimizer_config under the key 'mask'
        :param learning_rate_schedule: optional learning rate schedule, defaults to None
            if this is not None, it is added to (a copy of) optimizer_config under the key 'learning_rate'
            if this is a list or tuple, it is joined using optax.join_schedules with schedule_boundaries for boundaries
        :param schedule_boundaries: boundaries for optax.join_schedules
            only used if learning_rate_schedule is a list or tuple
        :return: an optax optimizer
        """
        optimizer_config = dict(optimizer_config)
        if learning_rate_schedule is not None:
            if isinstance(learning_rate_schedule, (list, tuple)):
                learning_rate_schedule = optax.join_schedules(learning_rate_schedule, schedule_boundaries)
            optimizer_config['learning_rate'] = learning_rate_schedule
        if optimizer_mask is not None:
            optimizer_config['mask'] = optimizer_mask
        return optimizer_type(**optimizer_config)


def make_inr_train_step_function(
        loss_evaluator:LossEvaluator,
        sampler:Sampler, 
        optimizer:optax.GradientTransformation, 
        ):
    """make_inr_train_step_function
    Make an INR train_step function to train an INR using a given loss_evaluator, sampler, and optimizer (and optionally state_update_function)

    :param loss_evaluator: a function that takes two arguments and has a scalar output
        the first argument should be an eqx.Module representing the INR that is to be trained
        the second argument should be a pytree containing input data coming from the sampler
           for example: an array of locations to evaluate the inr and a target function at.
        the third argument should be an optional eqx.nn.State
    :param sampler: sampler for input data for calculating the loss, e.g. 
        should have the following signature:
        :param key: jax.Array functioning as a prng key
        :return: a pytree containing the information needed to calculate the loss
            e.g. a jax.Array of coordinates
    :param optimizer: optax optimizer

    :return: train_step function, evaluate_loss function
        train_step has the follwing signature:
            :param inr: inr model to be optimized
            :param optimizer_state: state of the used (optax) optimizer
            :param key: key for prng
            :param state: optional state which is passed to the loss_evaluator (which can pass it to the inr) 
            :return: updated inr, updated optimizer_state, loss for this step, updated state

    NB the train_step function is decorated with eqx.filter_jit, so it can be jit-ed at the highest level.
    This means that ideally, the loss_evaluator and sampler should not be jit-ed.
    But they should all be jit-able! 
    """

    @eqx.filter_jit  # this is likely the highest level at which we can reasonably jit
    def train_step(inr:eqx.Module, optimizer_state:optax.OptState, key:jax.Array, state:Optional[eqx.nn.State]=None):
        """take one train step with the inr model and optimizer

        :param inr: inr model to be optimized
        :param optimizer_state: state of the used (optax) optimizer
        :param key: key for prng
        :param state: optional state passed to the loss_evaluator if not None
        :return: updated inr, updated optimizer_state, loss for this step 

        NB this function is decorated with eqx.filter_jit
        """
        locations = sampler(key)  # think of an array of coordinates, although this can in general be any PyTree containing the relevant information for the loss_evaluator
        (loss, state), grad = eqx.filter_value_and_grad(loss_evaluator, has_aux=True)(inr, locations, state)
        grad = jax.tree.map(
            lambda x: jnp.conjugate(x) if jnp.iscomplexobj(x) else x,
            grad
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state, inr)
        inr = eqx.apply_updates(inr, updates)
        return inr, optimizer_state, loss, state
    return train_step

def make_sampler_free_train_step(
        loss_evaluator,
        optimizer
        ):
    @eqx.filter_jit
    def train_step(inr: eqx.Module, batch:PyTree, optimizer_state:optax.OptState, state:Optional[eqx.nn.State]=None):
        (loss, state), grad = eqx.filter_value_and_grad(loss_evaluator, has_aux=True)(inr, batch, state)
        grad = jax.tree.map(
            lambda x: jnp.conjugate(x) if jnp.iscomplexobj(x) else x,
            grad
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state, inr)
        inr = eqx.apply_updates(inr, updates)
        return inr, optimizer_state, loss, state
    return train_step


def initialize_state(inr: eqx.Module)->tuple[eqx.Module, Optional[eqx.nn.State]]:
    """ 
    Initialize the state for a model
    If the model does not require a state, the state is set to None
    :param inr: eqx.Module *instance* for which a state is to be created
    :return: inr with initial state removed, initial state for that inr
    """
    # first we ensure that the markers in the state indices in the model are set correctly
    # and we keep track of whether we actually have any state indices
    # because if not, we just give None for a state

    leaves, treedef = jax.tree.flatten(inr, is_leaf=lambda x: isinstance(x, eqx.nn.StateIndex))
    counter = 0
    new_leaves = []

    for leaf in leaves:
        if isinstance(leaf, eqx.nn.StateIndex):
            leaf = eqx.nn.StateIndex(leaf.init)
            object.__setattr__(leaf, "marker", counter)
            counter += 1
        new_leaves.append(leaf)

    inr = jax.tree.unflatten(treedef, new_leaves)
    state = None

    if counter > 0:
        state = eqx.nn.State(inr)
        inr = eqx.nn.delete_init_state(inr)
    
    return inr, state


def train_inr(
        inr:eqx.Module,
        loss_evaluator:LossEvaluator,
        sampler:Sampler, 
        optimizer:optax.GradientTransformation,
        steps: int,
        key: jax.Array,
        use_wandb: bool,
        after_step_callback: Union[None, Callable, Callback] = None,
        after_training_callback: Union[None, Callable, eqx.Module] = None,
        optimizer_state: Union[None, optax.OptState] = None,
        state_initialization_function: Callable[[eqx.Module], tuple[eqx.Module, eqx.nn.State]] = initialize_state,
        state: Optional[eqx.nn.State] = None,
        ):
    """train_inr train an inr

    :param inr: inr to be trained
    :param loss_evaluator: a function that takes two arguments and has a scalar output
        the first argument should be an eqx.Module representing the INR that is to be trained
        the second argument should be a pytree containing input data coming from the sampler
           for example: an array of locations to evaluate the inr and a target function at.
        the third argument should be an optional eqx.nn.State
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
        :param state: state used by the inr
        :param optimizer_state: state of the optimizer
        :return: None (output is ignored)
        This can e.g. be used to plot the inr every so often
    :param after_training_callback: optional function or Callback to be called after training
        if None is provided, the losses will be plotted instead.
        if a Callable is provided, it will be passed the following key-word arguments:
            :param losses: list of loss values during training
            :param inr: final inr after training
            :param state: state after training
            :param optimizer_state: optimizer_state after training
            :param loss_evaluator: a function that takes two arguments and has a scalar output
                the first argument should be an eqx.Module representing the INR that is to be trained
                the second argument should be a pytree containing input data coming from the sampler
                for example: an array of locations to evaluate the inr and a target function at.
                the third argument should be an optional eqx.nn.State
            :param sampler: sampler used for probing target_function and inr during training
            :param key: key for PRNG
        its output will be returned as additional_output
    :param optimizer_state: optional optax.OptState in case you want to continue a training process instead of starting a new one
        If None is provided, the state is initialized using optimizer.init
    :param state_initialization_function: callable to create a state from in INR (eqx.Module instance)
        default function returns None if the INR doesn't require a state and otherwise creates a state from the init values of the state leafs of the INR
        if another Callable is provided it should have the call signature Callable[[eqx.Module], tuple[eqx.Module, eqx.nn.State]]
        will not be used if a `state` other than None is provided 
    :param state: opotional initial state. If None is provided, state_initialization_function will be used to create an initial state.
    :return: inr, losses, optimizer_state, state, loss_evaluator, additional_output

    NB the train_step function is decorated with eqx.filter_jit, so it can be jit-ed at the highest level.
    This means that ideally, the loss_function, target_function, and sampler should not be jit-ed.
    But they should all be jit-able! 
    """
    train_step = make_inr_train_step_function(
        loss_evaluator=loss_evaluator,
        sampler=sampler,
        optimizer=optimizer,
    )
    key_gen = key_generator(key)  # upon next, this just splits the key it has internally stored, and returns the second half of that split
    
    if state is None:
        inr, state = state_initialization_function(inr)
    
    if optimizer_state is None:
        optimizer_state = optimizer.init(eqx.filter(inr, eqx.is_array))

    losses = np.empty(steps, dtype=np.float32)

    for step in range(steps):  # the actual training loop
        inr, optimizer_state, loss, state = train_step(inr, optimizer_state, next(key_gen), state)
        losses[step] = loss
        after_step_callback(step, loss, inr, state, optimizer_state)

    if after_training_callback is None:
        if np.all(losses>0):
            plt.yscale("log")
        plt.plot(losses)
        plt.show()
        additional_output = None
    else:
        additional_output = after_training_callback(
            losses=losses, 
            inr=inr, 
            state=state,
            optimizer_state=optimizer_state,
            loss_evaluator=loss_evaluator, 
            sampler=sampler, 
            key=next(key_gen)
            )

    if use_wandb:  # upload the model to weights and biases
        import wandb
        import tempfile
        # first save the model to a temporary file
        with tempfile.NamedTemporaryFile() as f:
            eqx.tree_serialise_leaves(f, inr)
            wandb.log_model(path=f.name, name='inr.eqx')
    return inr, losses, optimizer_state, state, loss_evaluator, additional_output


def train_inr_with_dataloader(
        inr: eqx.Module,
        loss_evaluator: LossEvaluator,
        dataloader: DataLoader,
        optimizer: optax.GradientTransformation,
        steps: int,
        use_wandb: bool,
        after_step_callback: Union[None, Callable, Callback] = None,
        after_training_callback: Union[None, Callable, eqx.Module] = None,
        optimizer_state: Union[None, optax.OptState] = None,
        state_initialization_function: Callable[[eqx.Module], tuple[eqx.Module, eqx.nn.State]] = initialize_state,
        state: Optional[eqx.nn.State] = None,
        ):
    train_step = make_sampler_free_train_step(loss_evaluator=loss_evaluator, optimizer=optimizer)
    if state is None:
        inr, state = state_initialization_function(inr)
    
    if optimizer_state is None:
        optimizer_state = optimizer.init(eqx.filter(inr, eqx.is_array))

    losses = np.empty(steps, dtype=np.float32)

    for step, batch in zip(range(steps), iter(dataloader)):  # the actual training loop
        inr, optimizer_state, loss, state = train_step(inr, batch, optimizer_state, state)
        losses[step] = loss
        after_step_callback(step, loss, inr, state, optimizer_state)

    if after_training_callback is None:
        if np.all(losses>0):
            plt.yscale("log")
        plt.plot(losses)
        plt.show()
        additional_output = None
    else:
        additional_output = after_training_callback(
            losses=losses, 
            inr=inr, 
            state=state,
            optimizer_state=optimizer_state,
            loss_evaluator=loss_evaluator, 
            dataloader=dataloader, 
            )

    if use_wandb:  # upload the model to weights and biases
        import wandb
        import tempfile
        # first save the model to a temporary file
        with tempfile.NamedTemporaryFile() as f:
            eqx.tree_serialise_leaves(f, inr)
            wandb.log_model(path=f.name, name='inr.eqx')
    return inr, losses, optimizer_state, state, loss_evaluator, additional_output


# ########################################################################################################################################
# the following is for training multiple inrs in parallel

def make_scannable_step_function(
        loss_evaluator:LossEvaluator,
        sampler:Sampler, 
        optimizer:optax.GradientTransformation, 
        inr_static:eqx.Module,
        ):
    def train_step(carry, key):
        inr_weights, optimizer_state, state = carry
        inr = eqx.combine(inr_weights, inr_static)
        locations = sampler(key)
        (loss, state), grad = eqx.filter_value_and_grad(loss_evaluator, has_aux=True)(inr, locations, state)
        grad = jax.tree.map(
            lambda x: jnp.conjugate(x) if jnp.iscomplexobj(x) else x,
            grad
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state, inr)
        inr = eqx.apply_updates(inr, updates)
        inr_weights, _ = eqx.partition(inr, eqx.is_array)

        return (inr_weights, optimizer_state, state), loss
    return train_step

def train_inr_scan(
        inr: eqx.Module,
        key:jax.Array,
        steps:int,
        loss_evaluator:LossEvaluator,
        sampler:Sampler, 
        optimizer:optax.GradientTransformation, 
        state_initialization_function: Callable[[eqx.Module], tuple[eqx.Module, eqx.nn.State]] = initialize_state
        ):
    # prepare everything
    inr, state = state_initialization_function(inr)
    inr_weights, inr_static = eqx.partition(inr, eqx.is_array)
    optimizer_state = optimizer.init(inr_weights)
    step_function = make_scannable_step_function(
        loss_evaluator=loss_evaluator,
        sampler=sampler,
        optimizer=optimizer,
        inr_static=inr_static
    )
    # train the model
    keys = jax.random.split(key, num=steps)
    (inr_weights, optimizer_state, state), losses = jax.lax.scan(step_function, (inr_weights, optimizer_state, state), keys)

    inr = eqx.combine(inr_weights, inr_static)
    return inr, optimizer_state, state, losses


#############################################################################################################################################
# The above turned out to be much faster for regular training as well, so let's do something similar for training with a dataloader (for NeRFs)

def make_sampler_free_train_cycle(
        loss_evaluator: LossEvaluator,
        optimizer: optax.GradientTransformation,
        inr_static: eqx.Module
        ):
    def train_step(carry, batch):
        inr_weights, optimizer_state, state = carry
        inr = eqx.combine(inr_weights, inr_static)
        (loss, state), grad = eqx.filter_value_and_grad(loss_evaluator, has_aux=True)(inr, batch, state)
        grad = jax.tree.map(
            lambda x: jnp.conjugate(x) if jnp.iscomplexobj(x) else x,
            grad
        )
        updates, optimizer_state = optimizer.update(grad, optimizer_state, inr)
        inr = eqx.apply_updates(inr, updates)
        inr_weights, _ = eqx.partition(inr, eqx.is_array)

        return (inr_weights, optimizer_state, state), loss
    
    @jax.jit
    def train_cycle(carry, batches):
        carry, losses = jax.lax.scan(train_step, carry, batches)
        return carry, losses
    return train_cycle

def train_with_dataloader_scan(
        inr: eqx.Module,
        loss_evaluator: LossEvaluator,
        dataloader: DataLoader,
        optimizer: optax.GradientTransformation,
        steps_per_cycle: int,
        num_cycles: int,
        use_wandb: bool,
        after_cycle_callback: Union[None, Callable, Callback] = None,
        after_training_callback: Union[None, Callable, eqx.Module] = None,
        optimizer_state: Union[None, optax.OptState] = None,
        state_initialization_function: Callable[[eqx.Module], tuple[eqx.Module, eqx.nn.State]] = initialize_state,
        state: Optional[eqx.nn.State] = None,
        ):
    if state is None:
        inr, state = state_initialization_function(inr)
    inr_weights, inr_static = eqx.partition(inr, eqx.is_array)
    
    if optimizer_state is None:
        optimizer_state = optimizer.init(inr_weights)
    
    train_cycle = make_sampler_free_train_cycle(
        loss_evaluator=loss_evaluator,
        optimizer=optimizer,
        inr_static=inr_static
    )

    losses = num_cycles * [None]

    carry = (inr_weights, optimizer_state, state)

    dataloader_iter = iter(dataloader)

    @jax.jit
    def stack_trees(trees):
        return jax.tree.map(
            lambda *x: jnp.stack(x, axis=0),
            *trees
        )
    # import time

    for cycle in range(num_cycles):
        # t0 = time.time()
        batches = [next(dataloader_iter) for _ in range(steps_per_cycle)]
        batches = stack_trees(batches)
        # jax.block_until_ready(batches)
        # t1 = time.time()
        carry, loss_array = train_cycle(carry, batches)
        losses[cycle] = loss_array
        # jax.block_until_ready(loss_array)
        # t2 = time.time()
        after_cycle_callback(cycle, loss_array.mean(), eqx.combine(carry[0], inr_static), carry[2], carry[1])
        # t3 = time.time()
        # print(f"collecting batches: {t1-t0}s,\ntraining: {t2-t1}s,\nafter_cycle_callback: {t3-t2}s\n")
    
    losses = np.asarray(jnp.stack(losses, axis=0).flatten())
    inr_weights, optimizer_state, state = carry
    inr = eqx.combine(inr_weights, inr_static)

    if after_training_callback is None:
        if np.all(losses>0):
            plt.yscale("log")
        plt.plot(losses)
        plt.show()
        additional_output = None
    else:
        additional_output = after_training_callback(
            losses=losses, 
            inr=inr, 
            state=state,
            optimizer_state=optimizer_state,
            loss_evaluator=loss_evaluator, 
            dataloader=dataloader
            )

    if use_wandb:  # upload the model to weights and biases
        import wandb
        import tempfile
        # first save the model to a temporary file
        with tempfile.NamedTemporaryFile() as f:
            eqx.tree_serialise_leaves(f, inr)
            wandb.log_model(path=f.name, name='inr.eqx')
    return inr, losses, optimizer_state, state, loss_evaluator, additional_output
