import functools

import jax
import equinox as eqx

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