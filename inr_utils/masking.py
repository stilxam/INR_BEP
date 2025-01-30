import jax
import equinox as eqx

def array_mask(tree):
    return jax.tree.map(eqx.is_array, tree, is_leaf=lambda x: x is None or eqx.is_array(x))
