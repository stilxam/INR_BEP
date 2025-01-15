"""
Classes that define entire INR models, i.e. models that are composed of INRLayers.
The __init__ method of these classes typically takes entire layers or modules as arguments.
If you want to create instances of these classes from hyperparameters such as input size, hidden size, output size, etc.,
you can either use a from_config method when available, or provide INRLayer instances created from these hyperparameters by their from_config methods.
"""
from abc import ABC
from dataclasses import InitVar
from typing import Callable, Optional, Union, Any, Tuple, List
from functools import partial, wraps
import warnings
import abc

import jax
from jax import numpy as jnp
from jax import random, lax
import equinox as eqx
from jax._src.numpy.linalg import outer

from model_components.inr_layers import INRLayer, Linear, PositionalEncodingLayer, SirenLayer
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

    net: MLPINR

    def is_stateful(self):
        return self.net.is_stateful()

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

        return cls(blocks)

    def __call__(self, x, h:Optional[jax.Array]=None, state:Optional[eqx.nn.State]=None, *, key: Optional[jax.Array]=None) -> Tuple[jax.Array, jax.Array]:
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
        inp = h if h is not None else x
        if self.is_stateful():
            h_new, new_state = self.net(inp, state)
            return jnp.concatenate([x, h_new], axis=-1), state
        h_new = self.net(inp)
        return x, jnp.concatenate([x, h_new], axis=-1)

class NeRFComponent(INRModule):
    blocks: List[Union[INRLayer, NeRFBlock]]
    condition: Optional[List[Union[INRLayer, NeRFBlock]]]
    to_sigma: INRLayer
    to_rgb: INRLayer

    @classmethod
    def from_config(
        cls,
        in_size: Tuple[int, int] = (3,2),
        out_size: Tuple[int, int] = (1,3),
        bottle_size: int = 256,
        block_length: int = 4,
        block_width: int = 512,
        num_blocks: int = 2,
        condition_length: Optional[int] = None,
        condition_width: Optional[int] = None,
        layer_type: type[INRLayer] = SirenLayer,
        activation_kwargs: dict = {"s0": 5},
        key: jax.Array= random.PRNGKey(0),
        initialization_scheme: Optional[Callable] = None,
        initialization_scheme_kwargs: Optional[Callable] = None,
        positional_encoding_layer: Optional[PositionalEncodingLayer] = None,
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
            first_layer = positional_encoding_layer
            first_hidden_size = positional_encoding_layer.out_size(in_size=pos_coords)
        else:
            first_layer = initialization_scheme(
                in_size=pos_coords,
                out_size=block_width,
                num_splits=num_splits,
                key=next(key_gen),
                is_first_layer=True,
                **activation_kwargs
            )
            first_hidden_size = block_width



        blocks = [first_layer]
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
                    in_size=block_width,
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
                        in_size=block_width,
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
                    in_size=block_width,
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
            if positional_encoding_layer is not None:
                condition_first_layer = positional_encoding_layer
                condition_first_hidden_size = positional_encoding_layer.out_size(in_size=pos_coords)
            else:
                condition_first_layer = initialization_scheme(
                    in_size=pos_coords,
                    out_size=block_width,
                    num_splits=num_splits,
                    key=next(key_gen),
                    is_first_layer=True,
                    **activation_kwargs
                )
                condition_first_hidden_size = block_width
            condition = [condition_first_layer]
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
                # NeRFBlock.from_config(
                    # in_size=bottle_size+condition_first_hidden_size,
                    # out_size = bottle_size,
                    # hidden_size=condition_width,
                    # block_len=condition_length,
                    # layer_type=layer_type,
                    # activation_kwargs=activation_kwargs,
                    # initialization_scheme=initialization_scheme,
                    # initialization_scheme_kwargs=initialization_scheme_kwargs,
                    # key=next(key_gen)
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

        return cls(
            blocks=blocks,
            condition=condition,
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
            key (Optional[jax.Array]): Optional random number generator key.

        Returns:
            Tuple[jax.Array, jax.Array]: Output RGB and sigma values.
        """
        # TODO: Implement stateful layers
        h = position
        for component in self.blocks:
            h = component(h)

        raw_sigma = self.to_sigma(h)

        if self.condition is not None:
            h_view = self.condition[0](view_angle)
            h = jnp.concat([h, h_view], axis=-1)
            h = self.condition[1](h)

        raw_rgb = self.to_rgb(h)

        return raw_rgb, raw_sigma








class NeRFModel(INRModule):
    """
    A NeRF model that is composed of two NeRFComponents, one for the coarse nerf and one for the fine nerf.
    implementations strongly inspired by Google Research's JAX NeRF implementation
    https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/models.py

    """
    coarse_mlp: type[NeRFComponent]
    fine_mlp: type[NeRFComponent]
    condition: Union[bool, None]

    num_coarse_samples: int  # The number of samples for the coarse nerf.
    num_fine_samples: int  # The number of samples for the fine nerf.
    use_viewdirs: bool  # If True, use viewdirs as an input.
    near: float  # The distance to the near plane
    far: float  # The distance to the far plane
    noise_std: float  # The std dev of noise added to raw sigma.

    num_rgb_channels: int  # The number of RGB channels.
    num_sigma_channels: int  # The number of density channels.
    white_bkgd: bool  # If True, use a white background.
    min_deg_point: int  # The minimum degree of positional encoding for positions.
    max_deg_point: int  # The maximum degree of positional encoding for positions.
    deg_view: int  # The degree of positional encoding for viewdirs.
    lindisp: bool  # If True, sample linearly in disparity rather than in depth.
    legacy_posenc_order: bool  # Keep the same ordering as the original tf code.

    # activation_kwargs: Callable # i don't think we need to have it in this class as it is passed to the nerfcomponents

    # net_activation: Callable[..., Any]  # MLP activation
    rgb_activation: Callable = jax.nn.sigmoid
    sigma_activation: Callable = jax.nn.relu
    key: jax.Array

    @classmethod
    def from_config(cls,
                    num_coarse_samples: int,
                    num_fine_samples: int,
                    use_viewdirs: bool,
                    near: float,
                    far: float,
                    noise_std: float,
                    hidden_size: int,
                    net_depth: int,
                    condition_size: int,
                    net_depth_condition: Optional[int],
                    layer_type: type[INRLayer],
                    skip_layer: int,
                    num_rgb_channels: int,
                    num_sigma_channels: int,
                    white_bkgd: bool,
                    min_deg_point: int,
                    max_deg_point: int,
                    deg_view: int,
                    lindisp: bool,

                    rgb_activation: Callable[..., Any],
                    sigma_activation: Callable[..., Any],

                    activation_kwargs: dict,
                    key: jax.Array,
                    initialization_scheme: Optional[Callable] = None,
                    initialization_scheme_kwargs: Optional[dict] = None,
                    positional_encoding_layer: Optional[PositionalEncodingLayer] = None,
                    post_processor: Optional[Callable] = None,
                    ):
        """
        :param num_coarse_samples: int, the number of samples for the coarse nerf.
        :param num_fine_samples: int, the number of samples for the fine nerf.
        :param use_viewdirs: bool, if True, use viewdirs as an input.
        :param near: float, the distance to the near plane
        :param far: float, the distance to the far plane
        :param noise_std: float, the std dev of noise added to raw sigma.
        :param hidden_size: int, hidden size of network
        :param net_depth: int, number of layers
            NB this is including the input layer (which is possibly an encoding layer) and the output layer
        :param condition_size: int, size of the condition
        :param net_depth_condition: int, depth of the condition
        :param layer_type: type of the layers that is to be used
        :param skip_layer: int, number of layers to skip
        :param num_rgb_channels: int, the number of RGB channels.
        :param num_sigma_channels: int, the number of density channels.
        :param white_bkgd: bool, if True, use a white background.
        :param min_deg_point: int, the minimum degree of positional encoding for positions.
        :param max_deg_point: int, the maximum degree of positional encoding for positions.
        :param deg_view: int, the degree of positional encoding for viewdirs.
        :param lindisp: bool, if True, sample linearly in disparity rather than in depth.
        :param legacy_posenc_order: bool, keep the same ordering as the original tf code.
        :param rgb_activation: Callable[..., Any], activation function for rgb
        :param sigma_activation: Callable[..., Any], activation function for sigma
        :param activation_kwargs: dict, activation_kwargs to be passed to the layers
        :param key: jax.Array, key for random number generation
        :param initialization_scheme: (optional) callable that initializes layers
            it's call signature should be compatible with that of INRLayer.from_config (when the layer_type is set using functools.partial)
        :param initialization_scheme_kwargs: (optional) dict of kwargs to be passed to the initializaiton scheme (using functools.partial)
            NB only used if initialization_scheme is provided
        :param positional_encoding_layer: (optional) PositionalEncodingLayer instance to be used as the first layer
        :param post_processor: (optional) callable to be used on the output (by default: real_part, which takes the real part of a possibly complex array)
        :return: a NeRFModel object according to specification
        """

        x = jnp.exp(jnp.linspace(-90, 90, 1024))
        x = jnp.concatenate([-x[::-1], x], 0)
        rgb = rgb_activation(x)
        assert jnp.all(rgb >= 0) and jnp.all(rgb <= 1), "rgb_activation does not produce outputs in [0, 1]"

        sigma = sigma_activation(x)
        assert jnp.all(sigma >= 0), "sigma_activation does not produce non-negative outputs"

        if net_depth_condition is None:
            cls.condition = None
        else:
            cls.condition = True

        key_gen = key_generator(key=key)

        in_size = 5 if use_viewdirs else 3
        out_size = num_rgb_channels + num_sigma_channels

        cls.coarse_mlp = NeRFComponent.from_config(
            in_size=in_size,
            out_size=out_size,
            hidden_size=hidden_size,
            condition_size=condition_size,
            num_layers=net_depth,
            condition_depth=net_depth_condition,
            skip_layer=skip_layer,
            layer_type=layer_type,
            activation_kwargs=activation_kwargs,
            key=next(key_gen),
            initialization_scheme=initialization_scheme,
            initialization_scheme_kwargs=initialization_scheme_kwargs,
            positional_encoding_layer=positional_encoding_layer,
            num_splits=1,
            post_processor=post_processor,

        )
        cls.fine_mlp = NeRFComponent.from_config(

            in_size=in_size,
            out_size=out_size,
            hidden_size=hidden_size,
            condition_size=condition_size,
            num_layers=net_depth,
            condition_depth=net_depth_condition,
            skip_layer=skip_layer,
            layer_type=layer_type,
            activation_kwargs=activation_kwargs,
            key=next(key_gen),
            initialization_scheme=initialization_scheme,
            initialization_scheme_kwargs=initialization_scheme_kwargs,
            positional_encoding_layer=positional_encoding_layer,
            num_splits=1,
            post_processor=post_processor,
        )

        cls.num_coarse_samples = num_coarse_samples
        cls.num_fine_samples = num_fine_samples
        cls.use_viewdirs = use_viewdirs
        cls.near = near
        cls.far = far
        cls.noise_std = noise_std
        cls.num_rgb_channels = num_rgb_channels
        cls.num_sigma_channels = num_sigma_channels
        cls.white_bkgd = white_bkgd
        cls.min_deg_point = min_deg_point
        cls.max_deg_point = max_deg_point
        cls.deg_view = deg_view
        cls.lindisp = lindisp

        return cls

    def __call__(self, rays, randomized: bool = False):
        # Stratified sampling along rays

        key_gen = key_generator(key=self.key)
        z_vals, samples = self.sample_along_rays(
            next(key_gen),
            rays.origins,
            rays.directions,
            self.num_coarse_samples,
            self.near,
            self.far,
            randomized,
            self.lindisp,
        )
        # samples_enc = self.posenc(
        #     samples,
        #     self.min_deg_point,
        #     self.max_deg_point,
        #     self.legacy_posenc_order,
        # )

        # Point attribute predictions
        # if self.use_viewdirs:
        #     viewdirs_enc = self.posenc(
        #         rays.viewdirs,
        #         0,
        #         self.deg_view,
        #         self.legacy_posenc_order,
        #     )
        #     raw_rgb, raw_sigma = self.coarse_mlp(samples_enc, viewdirs_enc)
        # else:
        #     raw_rgb, raw_sigma = self.coarse_mlp(samples_enc)

        if self.use_viewdirs:
            input = jnp.concatenate([samples, rays.viewdirs], axis=-1)
        else:
            input = samples

        if self.condition:
            raw_rgb, raw_sigma = self.coarse_mlp(input, self.condition)
        elif self.condition is None:
            raw_rgb, raw_sigma = self.coarse_mlp(input, None)
        else:
            raise ValueError("condition must be either True or None")

        # Add noises to regularize the density predictions if needed
        raw_sigma = self.add_gaussian_noise(
            next(key_gen),
            raw_sigma,
            self.noise_std,
            randomized,
        )
        rgb = self.rgb_activation(raw_rgb)
        sigma = self.sigma_activation(raw_sigma)
        # Volumetric rendering.
        comp_rgb, disp, acc, weights = self.volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            rays.directions,
            white_bkgd=self.white_bkgd,
        )
        ret = [
            (comp_rgb, disp, acc),
        ]
        # Hierarchical sampling based on coarse predictions
        if self.num_fine_samples > 0:
            z_vals_mid = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
            z_vals, samples = self.sample_pdf(
                next(key_gen),
                z_vals_mid,
                weights[Ellipsis, 1:-1],
                rays.origins,
                rays.directions,
                z_vals,
                self.num_fine_samples,
                randomized,
            )
            samples_enc = self.posenc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
                self.legacy_posenc_order,
            )

            if self.use_viewdirs:
                raw_rgb, raw_sigma = self.fine_mlp(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_sigma = self.fine_mlp(samples_enc)

            raw_sigma = self.add_gaussian_noise(
                next(key_gen),
                raw_sigma,
                self.noise_std,
                randomized,
            )
            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)
            comp_rgb, disp, acc, unused_weights = self.volumetric_rendering(
                rgb,
                sigma,
                z_vals,
                rays.directions,
                white_bkgd=self.white_bkgd,
            )
            ret.append((comp_rgb, disp, acc))
        return ret

    @staticmethod
    def cast_rays(z_vals, origins, directions):
        """
        Cast rays through pixel positions.
        """
        return origins[Ellipsis, None, :] + z_vals[Ellipsis, None] * directions[Ellipsis, None, :]

    def sample_along_rays(self, key, origins, directions, num_samples, near, far, randomized, lindisp):
        """
        Sample along rays.

        :param key: jnp.ndarray(float32), [2,], random number generator.
        :param origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
        :param directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
        :param num_samples: int, the number of samples.
        :param near: float, the distance to the near plane.
        :param far: float, the distance to the far plane.
        :param randomized: bool, use randomized samples.
        :param lindisp: bool, sample linearly in disparity rather than in depth.

        :return: z_vals: jnp.ndarray(float32), [batch_size, num_samples].
        :return: coords: jnp.ndarray(float32), [batch_size, num_samples, 3].
        """
        batch_size = origins.shape[0]

        t_vals = jnp.linspace(0., 1., num_samples)
        if lindisp:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
        else:
            z_vals = near * (1. - t_vals) + far * t_vals

        if randomized:
            mids = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
            upper = jnp.concatenate([mids, z_vals[Ellipsis, -1:]], -1)
            lower = jnp.concatenate([z_vals[Ellipsis, :1], mids], -1)
            t_rand = random.uniform(key, [batch_size, num_samples])
            z_vals = lower + (upper - lower) * t_rand
        else:
            # Broadcast z_vals to make the returned shape consistent.
            z_vals = jnp.broadcast_to(z_vals[None, Ellipsis], [batch_size, num_samples])

        coords = self.cast_rays(z_vals, origins, directions)
        return z_vals, coords

    #
    # @staticmethod
    # def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    #     """
    #     Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].
    #
    #     Instead of computing [sin(x), cos(x)], we use the trig identity
    #     cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
    #
    #     :param x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    #     :param min_deg: int, the minimum (inclusive) degree of the encoding.
    #     :param max_deg: int, the maximum (exclusive) degree of the encoding.
    #     :param legacy_posenc_order: bool, keep the same ordering as the original tf code.
    #
    #     :return: jnp.ndarray, encoded variables.
    #     """
    #     if min_deg == max_deg:
    #         return x
    #     scales = jnp.array([2 ** i for i in range(min_deg, max_deg)])
    #     if legacy_posenc_order:
    #         xb = x[Ellipsis, None, :] * scales[:, None]
    #         four_feat = jnp.reshape(
    #             jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
    #             list(x.shape[:-1]) + [-1])
    #     else:
    #         xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
    #                          list(x.shape[:-1]) + [-1])
    #         four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    #     return jnp.concatenate([x] + [four_feat], axis=-1)

    @staticmethod
    def add_gaussian_noise(key, raw, noise_std, randomized):
        """Adds gaussian noise to `raw`, which can used to regularize it.

        :param key: jnp.ndarray(float32), [2,], random number generator.
        :param raw: jnp.ndarray(float32), arbitrary shape.
        :param noise_std: float, The standard deviation of the noise to be added.
        :param randomized: bool, add noise if randomized is True.

        :return: raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
        """
        if (noise_std is not None) and randomized:
            return raw + random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
        else:
            return raw

    @staticmethod
    def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd):
        """Volumetric Rendering Function.

        :param rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3].
        :param sigma: jnp.ndarray(float32), density, [batch_size, num_samples, 1].
        :param z_vals: jnp.ndarray(float32), [batch_size, num_samples].
        :param dirs: jnp.ndarray(float32), [batch_size, 3].
        :param white_bkgd: bool.

        :return: comp_rgb: jnp.ndarray(float32), [batch_size, 3].
        """
        eps = 1e-10
        dists = jnp.concatenate([
            z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1],
            jnp.broadcast_to(1e10, z_vals[Ellipsis, :1].shape)
        ], -1)
        dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)
        # Note that we're quietly turning sigma from [..., 0] to [...].
        alpha = 1.0 - jnp.exp(-sigma[Ellipsis, 0] * dists)
        accum_prod = jnp.concatenate([
            jnp.ones_like(alpha[Ellipsis, :1], alpha.dtype),
            jnp.cumprod(1.0 - alpha[Ellipsis, :-1] + eps, axis=-1)
        ],
            axis=-1)
        weights = alpha * accum_prod

        comp_rgb = (weights[Ellipsis, None] * rgb).sum(axis=-2)
        depth = (weights * z_vals).sum(axis=-1)
        acc = weights.sum(axis=-1)
        # Equivalent to (but slightly more efficient and stable than):
        #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
        inv_eps = 1 / eps
        disp = acc / depth
        disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
        if white_bkgd:
            comp_rgb = comp_rgb + (1. - acc[Ellipsis, None])
        return comp_rgb, disp, acc, weights

    @staticmethod
    def piecewise_constant_pdf(key, bins, weights, num_samples, randomized):
        """Piecewise-Constant PDF sampling.

        :param key: jnp.ndarray(float32), [2,], random number generator.
        :param bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
        :param weights: jnp.ndarray(float32), [batch_size, num_bins].
        :param num_samples: int, the number of samples.
        :param randomized: bool, use randomized samples.

        :return: z_samples: jnp.ndarray(float32), [batch_size, num_samples].
        """
        # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
        # avoids NaNs when the input is zeros or small, but has no effect otherwise.
        eps = 1e-5
        weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
        padding = jnp.maximum(0, eps - weight_sum)
        weights += padding / weights.shape[-1]
        weight_sum += padding

        # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
        # starts with exactly 0 and ends with exactly 1.
        pdf = weights / weight_sum
        cdf = jnp.minimum(1, jnp.cumsum(pdf[Ellipsis, :-1], axis=-1))
        cdf = jnp.concatenate([
            jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf,
            jnp.ones(list(cdf.shape[:-1]) + [1])
        ],
            axis=-1)

        # Draw uniform samples.
        if randomized:
            # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
            u = random.uniform(key, list(cdf.shape[:-1]) + [num_samples])
        else:
            # Match the behavior of random.uniform() by spanning [0, 1-eps].
            u = jnp.linspace(0., 1. - jnp.finfo('float32').eps, num_samples)
            u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

        # Identify the location in `cdf` that corresponds to a random sample.
        # The final `True` index in `mask` will be the start of the sampled interval.
        mask = u[Ellipsis, None, :] >= cdf[Ellipsis, :, None]

        def find_interval(x):
            # Grab the value where `mask` switches from True to False, and vice versa.
            # This approach takes advantage of the fact that `x` is sorted.
            x0 = jnp.max(jnp.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), -2)
            x1 = jnp.min(jnp.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), -2)
            return x0, x1

        bins_g0, bins_g1 = find_interval(bins)
        cdf_g0, cdf_g1 = find_interval(cdf)

        t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        samples = bins_g0 + t * (bins_g1 - bins_g0)

        # Prevent gradient from backprop-ing through `samples`.
        return lax.stop_gradient(samples)

    @staticmethod
    def sample_pdf(self, key, bins, weights, origins, directions, z_vals, num_samples,
                   randomized):
        """Hierarchical sampling.

        Args:
          key: jnp.ndarray(float32), [2,], random number generator.
          bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
          weights: jnp.ndarray(float32), [batch_size, num_bins].
          origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
          directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
          z_vals: jnp.ndarray(float32), [batch_size, num_coarse_samples].
          num_samples: int, the number of samples.
          randomized: bool, use randomized samples.

        Returns:
          z_vals: jnp.ndarray(float32),
            [batch_size, num_coarse_samples + num_fine_samples].
          points: jnp.ndarray(float32),
            [batch_size, num_coarse_samples + num_fine_samples, 3].
        """
        z_samples = self.piecewise_constant_pdf(key, bins, weights, num_samples,
                                                randomized)
        # Compute united z_vals and sample points
        z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
        coords = self.cast_rays(z_vals, origins, directions)
        return z_vals, coords
