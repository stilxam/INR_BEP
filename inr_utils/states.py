import functools

import jax
from jax import numpy as jnp

from model_components.inr_layers import IntegerLatticeEncoding


def copy_state(state):
    return jax.tree.map(lambda x: x, state)

def handle_state(inr, state):
    """ 
    make it so that we can pretend the inr doesn't need or return a state
    """
    requires_state = hasattr(inr, 'is_stateful') and inr.is_stateful()
    if state is None and not requires_state:
        return inr
    
    @functools.wraps(inr)
    def wrapped_inr(x):
        out = inr(x, copy_state(state))  # copy the state because if we re-use an old state, we get an error
        if isinstance(out, tuple):
            return out[0]
        return out
    return wrapped_inr

def update_ile( state, inr, nr_increments):
    # this function (method) is called during the __call__ of a LossEvaluator
    # it is called on a state and an inr that's being trained

    def _is_relevant_layer(leaf):
        return isinstance(leaf, IntegerLatticeEncoding)
    # we want to update each leaf of the inr that is of this class

    new_state = state
    for leaf in jax.tree.leaves(inr, is_leaf=_is_relevant_layer):
        if _is_relevant_layer(leaf):
            # update the state for this leaf
            current_alpha = new_state.get(leaf._alpha_state_index)
            init_alpha = new_state.get(leaf._min_alpha_state_index)
            max_alpha = new_state.get(leaf._max_alpha_state_index)

            increment_size = (max_alpha - init_alpha) / nr_increments
            current_alpha = jnp.minimum(current_alpha + increment_size, max_alpha)

            new_state = new_state.set(leaf._alpha_state_index, current_alpha)

    return new_state
