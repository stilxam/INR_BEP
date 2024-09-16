""" 
Classes that define entire INR models, i.e. models that are composed of INRLayers.
The __init__ method of these classes typically takes entire layers or modules as arguments.
If you want to create instances of these classes from hyperparameters such as input size, hidden size, output size, etc.,
you can either use a from_config method when available, or provide INRLayer instances created from these hyperparameters by their from_config methods. 
"""
import jax
import equinox as eqx

from model_components.inr_layers import INRLayer, Linear
from model_components import auxiliary as aux

from typing import Callable, Optional, Union
from common_jax_utils import key_generator


class INRModule(eqx.Module):
    @classmethod
    def __subclasshook__(cls, maybe_subclass):
        if issubclass(maybe_subclass, INRLayer):
            return True
        return NotImplemented


class MLPINR(INRModule):
    """MLPINR 
    An MLP with INRLayers
    """

    input_layer: INRLayer
    hidden_layers: list[INRLayer]
    output_layer: Linear
    post_processor: Callable=aux.real_part

    @classmethod
    def from_config(
        cls, 
        in_size:int, 
        out_size:int, 
        hidden_size:int, 
        num_layers:Optional[int], 
        layer_type:Union[type[INRLayer], list[type[INRLayer]], tuple[type[INRLayer]]], 
        activation_kwargs:Union[dict, list[dict], tuple[dict]], 
        num_splits:Union[int, list[int], tuple[int]], 
        use_complex:bool, 
        key:jax.Array,
        post_processor:Callable=aux.real_part,
        ):
        """from_config create an instance based on hyper parameters
        note that the __init__ is automatically generated due to eqx.Module classes being data classes

        :param in_size: input size of network
        :param out_size: output size of network
        :param hidden_size: hidden size of network
        :param num_layers: number of layers (optional)
            if layer_type is a list or tuple of INRLayer types, num_layers is ignored
            otherwise it is required
        :param layer_type: types of the layers to be used
            if this is a type instance (a class), the number of layers should be provided as num_layers
            if this is an iterable of type instances, one layer will be constructed for each element of the iterable
        :param activation_kwargs: activation_kwargs to be passed to the layers. 
            if this is a dictionary, the same dictionary will be passed to each layer
            if this is a list or tuple of dictionaries, the dictionaries will be provided to the corresponding layers indexwise
                note that if the number of layers is dictated by num_layers, the length of this list or tuple should be num_layers-1
        :param use_complex: whether to use complex numbers in the hidden layers
        :param key: key for random number generation
        :param post_processor: callable to be used on the output (by default: real_part, which takes the real part of a possibly complex array)
        :raises ValueError: if the number of activation_kwargs is incompatible with the number of layers
        :return: an MLPINR object according to specification
        """
        key_gen = key_generator(key)

        if isinstance(layer_type, (list, tuple)):
            num_layers = len(layer_type)+1  # simply overwrite num_layers
            #layer_type = layer_type + type(layer_type)((Linear,))
        else:
            layer_type = (num_layers-1)*[layer_type]# + [Linear] 

        if not isinstance(activation_kwargs, (list, tuple)):
            activation_kwargs = (num_layers-1)*[activation_kwargs]
        elif len(activation_kwargs) != num_layers-1:
            raise ValueError(f"When providing a {type(activation_kwargs)} of values for activation_kwargs, the length of this {type(activation_kwargs)} should be num_layers-1. Got {len(activation_kwargs)=} but {num_layers-1=}.")
        
        if not isinstance(num_splits, (tuple, list)):
            num_splits = (num_layers-1)*[num_splits]
        elif len(num_splits) != num_layers-1:
            raise ValueError(f"When providing a {type(num_splits)} of values for num_splits, the length of this {type(num_splits)} should be num_layers-1. Got {len(num_splits)=} but {num_layers-1=}.")
        
        if use_complex:
            def from_config(l_type:type[INRLayer], *args, **kwargs):
                return l_type.complex_from_config(*args, **kwargs)
        else:
            def from_config(l_type:type[INRLayer], *args, **kwargs):
                return l_type.from_config(*args, **kwargs)

        input_layer = from_config(
            layer_type[0],
            in_size=in_size,
            out_size=hidden_size,
            num_splits=num_splits[0],
            key=next(key_gen),
            is_first_layer=True,
            **activation_kwargs[0]
        )
        layers = [
            from_config(
                lt,
                in_size=hidden_size,
                out_size=hidden_size,
                num_splits=ns,
                key=next(key_gen),
                is_first_layer=False,
                **kwargs
            )
            for lt, ns, kwargs in zip(layer_type[1:], num_splits[1:], activation_kwargs[1:])
        ]
        output_layer = from_config(
            Linear,
            in_size=hidden_size,
            out_size=out_size,
            key=next(key_gen),
            is_first_layer=False
        )
        return cls(input_layer, layers, output_layer, post_processor)

    def __call__(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.post_processor(self.output_layer(x))

class CombinedINR(INRModule):
    terms: tuple[INRModule]
    post_processor: Callable = aux.real_part
    """ 
    An INR that is the sum of multiple INRs.
    """

    def __init__(self, *terms:INRModule, post_processor:Callable=aux.real_part):
        self.terms = terms
        self.post_processor = post_processor

    def __call__(self, x):
        return self.post_processor(sum(term(x) for term in self.terms))
