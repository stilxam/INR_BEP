""" 
Classes that define entire INR models, i.e. models that are composed of INRLayers.
The __init__ method of these classes typically takes entire layers or modules as arguments.
If you want to create instances of these classes from hyperparameters such as input size, hidden size, output size, etc.,
you can either use a from_config method when available, or provide INRLayer instances created from these hyperparameters by their from_config methods. 
"""
from typing import Callable, Optional, Union
from functools import partial, wraps
import warnings
import abc

import jax
import equinox as eqx

from model_components.inr_layers import INRLayer, Linear, PositionalEncodingLayer
from model_components import auxiliary as aux

from common_jax_utils import key_generator

def _deprecated(obj=None, new_name=None):
    if obj is None:
        return partial(_deprecated, new_name=new_name)
    new_str = f" Use {new_name} instead." if new_name is not None else ""
    @wraps(obj)
    def wrapper(*args, **kwargs):
        warnings.warn(f"Deprecation warning: {obj.__name__} is deprecated.{new_str}")
        return obj(*args, **kwargs)
    return wrapper


class INRModule(eqx.Module):
    
    @abc.abstractmethod
    def is_stateful(self)->bool:
        pass


class MLPINR(eqx.nn.Sequential, INRModule):
    """MLPINR 
    Mostly just eqx.nn.Sequential, but with a
    class method to create an MLP of INR layers from 
    hyperparameters.
    """

    @classmethod
    def from_config(
        cls, 
        in_size:int, 
        out_size: int, 
        hidden_size: int, 
        num_layers: int, 
        layer_type:type[INRLayer], 
        activation_kwargs:dict, 
        key:jax.Array,
        initialization_scheme:Optional[Callable]=None,
        initialization_scheme_kwargs:Optional[dict]=None,
        positional_encoding_layer:Optional[PositionalEncodingLayer]=None,
        num_splits=1,
        post_processor:Optional[Callable]=None,
        ):
        """ 
        :param in_size: input size of network
        :param out_size: output size of network
        :param hidden_size: hidden size of network
        :param num_layers: number of layers
            NB this is including the input layer (which is possibly an encoding layer) and the output layer
        :param layer_type: type of the layers that is to be used
        :param activation_kwargs: activation_kwargs to be passed to the layers
        :param key: key for random number generation
        :param initialization_scheme: (optional) callable that initializes layers
            it's call signature should be compatible with that of INRLayer.from_config (when the layer_type is set using functools.partial)
        :param initialization_scheme_kwargs: (optional) dict of kwargs to be passed to the initializaiton scheme (using functools.partial)
            NB only used if initialization_scheme is provided
        :param positional_encoding_layer: (optional) PositionalEncodingLayer instance to be used as the first layer
        :param num_splits: number of weights matrices (and bias vectors) to be used by the layers that support that.
        :param post_processor: callable to be used on the output (by default: real_part, which takes the real part of a possibly complex array)
        :return: an MLPINR object according to specification
        """
        key_gen = key_generator(key=key)
        initialization_scheme_kwargs = initialization_scheme_kwargs or {}
        if initialization_scheme is None:
            initialization_scheme = layer_type.from_config
        else:
            initialization_scheme = partial(initialization_scheme, layer_type=layer_type, **initialization_scheme_kwargs)

        if positional_encoding_layer is not None:
            first_layer = positional_encoding_layer
            first_hidden_size = positional_encoding_layer.out_size(in_size=in_size)
        else:
            first_layer = initialization_scheme(
                in_size=in_size,
                out_size=hidden_size,
                num_splits=num_splits,
                key=next(key_gen),
                is_first_layer=True,
                **activation_kwargs
            )
            first_hidden_size = hidden_size

        layers = [first_layer]

        if num_layers>2:
            layers.append(initialization_scheme(
                in_size=first_hidden_size, 
                out_size=hidden_size, 
                num_splits=num_splits, 
                key=next(key_gen), 
                is_first_layer=False, 
                **activation_kwargs
                ))
            for _ in range(num_layers-3):
                layers.append(initialization_scheme(
                    in_size=hidden_size, 
                    out_size=hidden_size, 
                    num_splits=num_splits, 
                    key=next(key_gen), 
                    is_first_layer=False, 
                    **activation_kwargs
                    ))
            out_layer = Linear.from_config(
                in_size=hidden_size,
                out_size=out_size,
                num_splits=1,
                key=next(key_gen), 
                is_first_layer=False
            )
        else:
            out_layer = Linear.from_config(
                in_size=hidden_size,
                out_size=out_size,
                num_splits=1,
                key=next(key_gen), 
                is_first_layer=False
            )
        layers.append(out_layer)
        if post_processor is not None:
            if isinstance(post_processor, eqx.Module):
                layers.append(post_processor)
            else:
                layers.append(eqx.nn.Lambda(post_processor))
        return cls(layers)
            


class CombinedINR(INRModule):
    terms: tuple[INRModule]
    post_processor: Callable = aux.real_part
    """ 
    An INR that is the sum of multiple INRs.
    """

    def __init__(self, *terms: INRModule, post_processor: Callable = aux.real_part):
        self.terms = terms
        self.post_processor = post_processor

    def is_stateful(self):
        return any(term.is_stateful() for term in self.terms)

    def __call__(self, x, state:Optional[eqx.nn.State]=None):
        is_stateful = self.is_stateful()

        if not is_stateful:
            return self.post_processor(sum(term(x) for term in self.terms))
        elif state is None:
            return self.post_processor(sum(term(x) for term in self.terms)), None
        
        out = 0
        for term in self.terms:
            if term.is_stateful():
                substate = state.substate(term)
                result, substate = term(x, substate)
                state = state.update(substate)
            else:
                result = term(x)
            out += result
        return self.post_processor(out), state


class NeRFComponent(INRModule):
    pass

class NeRFModel(INRModule):
    coarse_model: NeRFComponent
    fine_model: NeRFComponent
