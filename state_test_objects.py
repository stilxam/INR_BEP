from typing import Optional

import jax
from jax import numpy as jnp
import equinox as eqx

from model_components.inr_layers import PositionalEncodingLayer
from model_components.inr_modules import MLPINR


class CountingIdentity(PositionalEncodingLayer):
    _is_learnable = False
    state_index: eqx.nn.StateIndex

    def __init__(self):
        self._embedding_matrix = jnp.empty((), dtype=jnp.int32)
        init_state = jnp.zeros((), dtype=jnp.int32)
        self.state_index = eqx.nn.StateIndex(init_state)

    def is_stateful(self):
        return True

    def out_size(self, in_size):
        return in_size

    def __call__(self, x: jax.Array, state: eqx.nn.State, *, key: Optional[jax.Array] = None) -> tuple[
        jax.Array, eqx.nn.State]:
        return x, state


def _is_CountingIdentity(layer):
    # this is a dumb hack that works around some import mechanics in common_dl_utils
    return isinstance(layer, eqx.Module) and type(layer).__name__ == "CountingIdentity"


def counter_updater(state: eqx.nn.State, inr: MLPINR):
    for leaf in jax.tree.leaves(inr, is_leaf=_is_CountingIdentity):
        if _is_CountingIdentity(leaf):
            counter = state.get(leaf.state_index)
            state = state.set(leaf.state_index, counter + 1)
    return state


def after_training_callback(losses, inr, state, *args, **kwargs):
    print("Checking model and state for CountingIdentity layers")
    for leaf in jax.tree.leaves(inr, is_leaf=_is_CountingIdentity):
        if _is_CountingIdentity(leaf):
            counter = state.get(leaf.state_index)
            print(f"Found a CountingIdentity layer with counter value {counter} in final state after training.")
    return None
