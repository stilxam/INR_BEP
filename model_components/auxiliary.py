""" 
Module of auxiliary functions and classes for use in the model_components module.
Just a bunch of stuff that doesn't fit anywhere else and doesn't really warrant its own module.
"""
from typing import Union, TypeAlias, NewType, Callable
from types import GenericAlias
from functools import wraps

import jax
from jax import numpy as jnp


ANNOT = Union[type, GenericAlias, TypeAlias, NewType]


def value_error_on_all_but_allowed_keys(activation_kwargs:dict, *allowed_keys:Union[str, tuple[str, ANNOT]]):
    """ 
    Checks whether the keys of the activation_kwargs dict match the allowed_keys
    :param activation_kwargs: dictionary the keys of which should be checked against allowed_keys
    :param allowed_keys: sequence of strings determining the keys activation_kwargs should have
    :raises: ValueError if set(activation_kwargs) != set(allowed_keys)
    """
    # first filter out the possible type annotations
    allowed_keys = tuple(key[0] if isinstance(key, tuple) else key for key in allowed_keys)
    # make it a set so we can compare it to the set of activation_kwargs
    allowed_keys = set(allowed_keys)
    if not allowed_keys:
        if activation_kwargs:
            raise ValueError(f"No activation_kwargs allowed. Got {activation_kwargs}")
    if (used_keys := set(activation_kwargs)) != allowed_keys:
        raise ValueError(f"activation_kwargs should have keys {allowed_keys}. Got {used_keys} instead.")

def filter_allowed_keys_or_raise_value_error(activation_kwargs:dict, *required_keys:Union[str, tuple[str, ANNOT]]):
    """ 
    Checks that all required keys are present in activation_kwargs
    Raises a ValueError if any required keys are missing from activation_kwargs
    Returns a filtered version of activation_kwargs where only the required keys are present
    """
    required_keys = tuple(key[0] if isinstance(key, tuple) else key for key in required_keys)
    required_keys = set(required_keys)
    missing_keys = required_keys - set(activation_kwargs)
    if missing_keys:
        raise ValueError(f"Missing keys: {missing_keys}. Got {activation_kwargs} but need {required_keys}.")
    return {key: value for key, value in activation_kwargs.items() if key in required_keys}

def real_part(obj: Union[Callable, jax.Array]):
    """
    take the real part of a complex object
    
    if obj is an array, the real part of the array is returned (jnp.real(obj))
    if obj is a callable, a wrapper arround said callable is returned that transforms the output of obj through jnp.real
    """
    if not callable(obj):
        return jnp.real(obj)
    @wraps(obj)
    def real_func(x, *args, **kwargs):
        return jnp.real(obj(x, *args, **kwargs))
    return real_func

def imaginary_part(obj: Union[Callable, jax.Array]):
    """
    take the imaginary part of a complex object
    
    if obj is an array, the imaginary part of the array is returned (jnp.imag(obj))
    if obj is a callable, a wrapper arround said callable is returned that transforms the output of obj through jnp.imag
    """
    if not callable(obj):
        return jnp.imag(obj)
    @wraps(obj)
    def real_func(x, *args, **kwargs):
        return jnp.imag(obj(x, *args, **kwargs))
    return real_func

def scalar_from_array_output(array: jax.Array):
    return jnp.squeeze(array)