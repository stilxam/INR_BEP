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
from jax import numpy as jnp
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
    def is_stateful(self) -> bool:
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
            in_size: int,
            out_size: int,
            hidden_size: int,
            num_layers: int,
            layer_type: type[INRLayer],
            activation_kwargs: dict,
            key: jax.Array,
            initialization_scheme: Optional[Callable] = None,
            initialization_scheme_kwargs: Optional[dict] = None,
            positional_encoding_layer: Optional[PositionalEncodingLayer] = None,
            num_splits=1,
            post_processor: Optional[Callable] = None,
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
            initialization_scheme = partial(initialization_scheme, layer_type=layer_type,
                                            **initialization_scheme_kwargs)

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

        if num_layers > 2:
            layers.append(initialization_scheme(
                in_size=first_hidden_size,
                out_size=hidden_size,
                num_splits=num_splits,
                key=next(key_gen),
                is_first_layer=False,
                **activation_kwargs
            ))
            for _ in range(num_layers - 3):
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

    def __call__(self, x, state: Optional[eqx.nn.State] = None):
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


class NeRFBlock(INRModule):
    """
    A block in the NeRF architecture that consists of multiple layers.
    This block can be used to build the main network of the NeRFComponent.

    Attributes:
        net (MLPINR): The MLP network that forms the block.
    """

    net: eqx.Module

    def is_stateful(self):
        return False  # we don't have stateful layers that aren't positional encodings

    # TODO from_config: just basically copy from MLPINR for inputs, and that refer to that
    @classmethod
    def from_config(
            cls,
            in_size: int,
            out_size: int,
            hidden_size: int,
            block_len: int,
            layer_type: type[INRLayer],
            activation_kwargs: dict,
            key: jax.Array,
            initialization_scheme: Optional[Callable] = None,
            initialization_scheme_kwargs: Optional[dict] = None,
            num_splits=1,
    ):
        """
        Creates a NeRFBlock instance from the given configuration.

        Args:
            in_size (int): Input size of the block.
            out_size (int): Output size of the block.
            hidden_size (int): Hidden size of the block.
            block_len (int): Length of the block (number of layers).
            layer_type (type[INRLayer]): Type of the layers to be used.
            activation_kwargs (dict): Dictionary of activation function arguments.
            key (jax.Array): Random number generator key.
            initialization_scheme (Optional[Callable]): Callable for layer initialization.
            initialization_scheme_kwargs (Optional[dict]): Dictionary of arguments for the initialization scheme.
            num_splits (int): Number of weight matrices and bias vectors to be used by the layers that support that.

        Returns:
            NeRFBlock: An instance of NeRFBlock.
        """
        key_gen = key_generator(key)
        initialization_scheme_kwargs = initialization_scheme_kwargs or {}
        if initialization_scheme is None:
            initialization_scheme = layer_type.from_config
        else:
            initialization_scheme = partial(initialization_scheme, layer_type=layer_type,
                                            **initialization_scheme_kwargs)

        layers = [
            initialization_scheme(
                                in_size=in_size,
                                out_size=hidden_size,
                                num_splits=num_splits,
                                key=next(key_gen),
                                is_first_layer=True,
                                **activation_kwargs
                            )
        ]
        if block_len ==1:
            layers = [
                initialization_scheme(
                                in_size=in_size,
                                out_size=out_size,
                                num_splits=num_splits,
                                key=next(key_gen),
                                is_first_layer=True,
                                **activation_kwargs
                            )
            ]
        elif block_len==2:
               layers.append(initialization_scheme(
                   in_size=hidden_size,
                   out_size=out_size,
                   num_splits=num_splits,
                   key=next(key_gen),
                   is_first_layer=False,
                   **activation_kwargs
               ))
        if block_len>2:
            for _ in range(block_len- 2):
                layers.append(initialization_scheme(
                    in_size=hidden_size,
                    out_size=hidden_size,
                    num_splits=num_splits,
                    key=next(key_gen),
                    is_first_layer=False,
                    **activation_kwargs
                ))
            layers.append(initialization_scheme(
                in_size=hidden_size,
                out_size=out_size,
                num_splits=num_splits,
                key=next(key_gen),
                is_first_layer=False,
                **activation_kwargs
            ))

        return cls(eqx.nn.Sequential(layers))

    def __call__(self, x, h:Optional[jax.Array]=None, *, key: Optional[jax.Array]=None) -> jax.Array:
        """
        Forward pass through the NeRFBlock.

        Args:
            x (jax.Array): Input array.
            h (Optional[jax.Array]): Optional hidden state.
            state (Optional[eqx.nn.State]): Optional state for stateful layers.
            key (Optional[jax.Array]): Optional random number generator key.

        Returns:
            Tuple[jax.Array, jax.Array]: Output array and updated hidden state.
        """
        inp = x if h is None else jnp.concatenate([x, h], axis=-1)
        return self.net(inp)


class NeRFComponent(INRModule):
    block_pos_enc: Union[INRLayer, PositionalEncodingLayer]
    blocks: list[NeRFBlock]

    to_sigma: INRLayer
    to_rgb: INRLayer

    conditional_pos_enc: Optional[Union[INRLayer, PositionalEncodingLayer]] = None
    condition: Optional[eqx.Module] = None

    def is_stateful(self):
        return isinstance(self.block_pos_enc, PositionalEncodingLayer) and self.block_pos_enc.is_stateful()  # only the positional encodings use state in these experiments

    @classmethod
    def from_config(
        cls,
        in_size: tuple[int, int],
        out_size: tuple[int, int],
        bottle_size: int,
        block_length: int,
        block_width: int,
        num_blocks: int,
        condition_length: Optional[int],
        condition_width: Optional[int],
        layer_type: type[INRLayer],
        activation_kwargs: dict,
        key: jax.Array,
        initialization_scheme: Optional[Callable] = None,
        initialization_scheme_kwargs: Optional[Callable] = None,
        positional_encoding_layer: Optional[PositionalEncodingLayer] = None,
        direction_encoding_layer: Optional[PositionalEncodingLayer] = None,
        num_splits: int = 1,
        post_processor: Optional[Callable] = None
    ):
        """
        Creates a NeRFComponent instance from the given configuration.

        Args:
            in_size (Tuple[int, int]): Tuple representing the number of position coordinates and view angles.
            out_size (Tuple[int, int]): Tuple representing the number of sigma channels and RGB channels.
            bottle_size (int): Size of the bottleneck layer.
            block_length (int): Length of each block.
            block_width (int): Width of each block.
            num_blocks (int): Number of blocks in the network.
            condition_length (Optional[int]): Length of the condition layers.
            condition_width (Optional[int]): Width of the condition layers.
            layer_type (type[INRLayer]): Type of the layers to be used.
            activation_kwargs (dict): Dictionary of activation function arguments.
            key (jax.Array): Random number generator key.
            initialization_scheme (Optional[Callable]): Callable for layer initialization.
            initialization_scheme_kwargs (Optional[Callable]): Dictionary of arguments for the initialization scheme.
            positional_encoding_layer (Optional[PositionalEncodingLayer]): Positional encoding layer instance.
            num_splits (int): Number of weight matrices and bias vectors to be used by the layers that support that.
            post_processor (Optional[Callable]): Callable to be used on the output.

        Returns:
            NeRFComponent: An instance of NeRFComponent.
        """

        pos_coords, view_coords = in_size
        num_sigma, num_rgb = out_size
        key_gen = key_generator(key=key)

        initialization_scheme_kwargs = initialization_scheme_kwargs or {}
        if initialization_scheme is None:
            initialization_scheme = layer_type.from_config
        else:
            initialization_scheme = partial(initialization_scheme, layer_type=layer_type,
                                            **initialization_scheme_kwargs)

        if positional_encoding_layer is not None:
            block_pos_enc: PositionalEncodingLayer = positional_encoding_layer
            first_hidden_size = block_pos_enc.out_size(in_size=pos_coords)
        else:
            # block_pos_enc= initialization_scheme(
            #     in_size=pos_coords,
            #     out_size=block_width,
            #     num_splits=num_splits,
            #     key=next(key_gen),
            #     is_first_layer=True,
            #     **activation_kwargs
            # )
            # first_hidden_size = block_width
            def block_pos_enc(x):
                return x
            first_hidden_size = pos_coords



        blocks = []
        if num_blocks == 1:
            blocks.append(
                NeRFBlock.from_config(
                    in_size=first_hidden_size,
                    out_size = bottle_size,
                    hidden_size=block_width,
                    block_len=block_length,
                    layer_type=layer_type,
                    activation_kwargs=activation_kwargs,
                    initialization_scheme=initialization_scheme,
                    initialization_scheme_kwargs=initialization_scheme_kwargs,
                    key=next(key_gen)
                )
            )
        elif num_blocks ==2:
            blocks.append(
                NeRFBlock.from_config(
                    in_size=first_hidden_size,
                    out_size = block_width,
                    hidden_size=block_width,
                    block_len=block_length,
                    layer_type=layer_type,
                    activation_kwargs=activation_kwargs,
                    initialization_scheme=initialization_scheme,
                    initialization_scheme_kwargs=initialization_scheme_kwargs,
                    key=next(key_gen)
                )
            )
            blocks.append(
                NeRFBlock.from_config(
                    in_size=block_width + first_hidden_size,
                    out_size = bottle_size,
                    hidden_size=block_width,
                    block_len=block_length,
                    layer_type=layer_type,
                    activation_kwargs=activation_kwargs,
                    initialization_scheme=initialization_scheme,
                    initialization_scheme_kwargs=initialization_scheme_kwargs,
                    key=next(key_gen)
                )
            )
        elif num_blocks >=3:
            blocks.append(
                NeRFBlock.from_config(
                    in_size=first_hidden_size,
                    out_size = block_width,
                    hidden_size=block_width,
                    block_len=block_length,
                    layer_type=layer_type,
                    activation_kwargs=activation_kwargs,
                    initialization_scheme=initialization_scheme,
                    initialization_scheme_kwargs=initialization_scheme_kwargs,
                    key=next(key_gen)
                )
            )
            for i in range(num_blocks-2):
                blocks.append(
                    NeRFBlock.from_config(
                        in_size=block_width + first_hidden_size,
                        out_size = block_width,
                        hidden_size=block_width,
                        block_len=block_length,
                        layer_type=layer_type,
                        activation_kwargs=activation_kwargs,
                        initialization_scheme=initialization_scheme,
                        initialization_scheme_kwargs=initialization_scheme_kwargs,
                        key=next(key_gen)
                    )
                )

            blocks.append(
                NeRFBlock.from_config(
                    in_size=block_width + first_hidden_size,
                    out_size = bottle_size,
                    hidden_size=block_width,
                    block_len=block_length,
                    layer_type=layer_type,
                    activation_kwargs=activation_kwargs,
                    initialization_scheme=initialization_scheme,
                    initialization_scheme_kwargs=initialization_scheme_kwargs,
                    key=next(key_gen)
                )
            )

        condition = None
        if condition_width and condition_length is not None:
            if direction_encoding_layer is not None:
                conditional_pos_enc = direction_encoding_layer
                condition_first_hidden_size = conditional_pos_enc.out_size(in_size=pos_coords)
            else:
                def conditional_pos_enc(x):
                    return x
                condition_first_hidden_size = pos_coords
            condition = []
            condition.append(
                MLPINR.from_config(
                    in_size=bottle_size+condition_first_hidden_size,
                    out_size = bottle_size,
                    hidden_size=condition_width,
                    num_layers=condition_length,
                    layer_type=layer_type,
                    activation_kwargs=activation_kwargs,
                    key=next(key_gen)
                )
                # )
            )

        to_sigma = Linear.from_config(
            in_size=bottle_size,
            out_size=num_sigma,
            key=next(key_gen),
            is_first_layer=False
        )
        to_rgb = Linear.from_config(
                    in_size=bottle_size,
                    out_size=num_rgb,
                    key=next(key_gen),
                    is_first_layer=False
                )
        if condition is not None:
            return cls(
                block_pos_enc=block_pos_enc,
                conditional_pos_enc=conditional_pos_enc,
                blocks=eqx.nn.Sequential(blocks),
                condition=eqx.nn.Sequential(condition),
                to_sigma=to_sigma,
                to_rgb=to_rgb
            )
        else:
            return cls(
                block_pos_enc=block_pos_enc,
                blocks=eqx.nn.Sequential(blocks),
                to_sigma=to_sigma,
                to_rgb=to_rgb
            )

    def __call__(self, position, view_angle, state: Optional[eqx.nn.State]=None, *, key: Optional[jax.Array] = None):
        """
        Forward pass through the NeRFComponent.

        Args:
            position (jax.Array): Input position coordinates.
            view_angle (jax.Array): Input view angles.
            state (Optional[eqx.nn.State]): Optional state for stateful layers.
            key (Optional[jax.Array]): Optional random number generator key. Not used, just here for compatibility with Equinox API

        Returns:
            Tuple[jax.Array, jax.Array]: Output RGB and sigma values.
        """

        if self.is_stateful():
            encoding = self.block_pos_enc(position, state)
        encoding = self.block_pos_enc(position)
        h = None

        for block in self.blocks:
            h = block(encoding, h)

        raw_sigma = self.to_sigma(h)

        if self.condition is not None:
            h_view = self.conditional_pos_enc(view_angle)
            h = jnp.concatenate([h, h_view], axis=-1)
            h = self.condition(h)

        raw_rgb = self.to_rgb(h)

        if self.is_stateful():
            return raw_rgb, raw_sigma, state
        return raw_rgb, raw_sigma



class NeRF(INRModule):
    coarse_model: NeRFComponent
    fine_model: NeRFComponent
    
    @property
    def is_stateful(self):
        return self.coarse_model.is_stateful() or self.fine_model.is_stateful()
    
    def __call__(self, position, view_angle, state: Optional[eqx.nn.State]=None, *, key: Optional[jax.Array] = None):
        return {
            'coarse': self.coarse_model(position=position, view_angle=view_angle, state=state, key=key),
            'fine': self.fine_model(position=position, view_angle=view_angle, state=state, key=key)
        }

    @classmethod
    def from_config(cls,
        in_size: tuple[int, int],
        out_size: tuple[int, int],
        bottle_size: int,
        block_length: int,
        block_width: int,
        num_blocks: int,
        condition_length: int,
        condition_width: int,
        layer_type: type[INRLayer],
        activation_kwargs: dict,
        key: jax.Array,
        initialization_scheme: Optional[Callable] = None,
        initialization_scheme_kwargs: Optional[dict] = None,
        positional_encoding_layer: Optional[PositionalEncodingLayer] = None,
        direction_encoding_layer: Optional[PositionalEncodingLayer] = None,
        num_splits:int=1,
        post_processor: Optional[Callable] = None,
        shared_initialization: bool = False,
        ):
        if shared_initialization:
            key_coarse, key_fine = key, key
        else:
            key_coarse, key_fine = jax.random.split(key)
        coarse_model = NeRFComponent.from_config(
                in_size,
                out_size,
                bottle_size,
                block_length,
                block_width,
                num_blocks,
                condition_length,
                condition_width,
                layer_type,
                activation_kwargs,
                key_coarse,
                initialization_scheme,
                initialization_scheme_kwargs,
                positional_encoding_layer,
                direction_encoding_layer,
                num_splits,
                post_processor
        )
        fine_model = NeRFComponent.from_config(
                        in_size,
                        out_size,
                        bottle_size,
                        block_length,
                        block_width,
                        num_blocks,
                        condition_length,
                        condition_width,
                        layer_type,
                        activation_kwargs,
                        key_fine,
                        initialization_scheme,
                        initialization_scheme_kwargs,
                        positional_encoding_layer,
                        direction_encoding_layer,
                        num_splits,
                        post_processor
                )
        return cls(
            coarse_model=coarse_model,
            fine_model=fine_model,
        )

