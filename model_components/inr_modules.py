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

    net: eqx.Module

    @property
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

        return cls(eqx.nn.Sequential(layers))

    def __call__(self, x, h:Optional[jax.Array]=None, state:Optional[eqx.nn.State]=None, *, key: Optional[jax.Array]=None) -> Union[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, eqx.nn.State]]:
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
        if self.is_stateful:
            h_new, new_state = self.net(inp, state)
            return x, jnp.concatenate([x, h_new], axis=-1), new_state
        h_new = self.net(inp)
        return x, jnp.concatenate([x, h_new], axis=-1)


class NeRFComponent(INRModule):
    block_pos_enc: Union[INRLayer, PositionalEncodingLayer]
    blocks: eqx.Module

    to_sigma: INRLayer
    to_rgb: INRLayer

    conditional_pos_enc: Optional[Union[INRLayer, PositionalEncodingLayer]] = None
    condition: Optional[eqx.Module] = None

    @property
    def is_stateful(self):
        return self.blocks.is_stateful

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
            block_pos_enc = positional_encoding_layer.from_config(
                num_frequencies=pos_coords,
            )
            first_hidden_size = block_pos_enc.out_size(in_size=pos_coords)
        else:
            block_pos_enc= initialization_scheme(
                in_size=pos_coords,
                out_size=block_width,
                num_splits=num_splits,
                key=next(key_gen),
                is_first_layer=True,
                **activation_kwargs
            )
            first_hidden_size = block_width



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
                condition_pos_enc = positional_encoding_layer
                condition_first_hidden_size = positional_encoding_layer.out_size(in_size=pos_coords) # TODO: check if this is correct
            else:
                conditional_pos_enc = initialization_scheme(
                    in_size=pos_coords,
                    out_size=block_width,
                    num_splits=num_splits,
                    key=next(key_gen),
                    is_first_layer=True,
                    **activation_kwargs
                )
                condition_first_hidden_size = block_width
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
            key (Optional[jax.Array]): Optional random number generator key.

        Returns:
            Tuple[jax.Array, jax.Array]: Output RGB and sigma values.
        """
        # h = position
#
        # if self.block_pos_enc is not None:
            # h = self.block_pos_enc(h)
#
#
        # for component in self.blocks:
            # inp, h = component(h)
#
        # raw_sigma = self.to_sigma(h)
#
        # if self.condition is not None:
            # h_view = self.conditional_pos_enc(view_angle)
            # h = jnp.concat([h, h_view], axis=-1)
            # h = self.condition[1](h)
#
        # raw_rgb = self.to_rgb(h)
#
        # return raw_rgb, raw_sigma
        #
        h = position

        
        h = self.block_pos_enc(h, key=key)

        new_state = state

        for component in self.blocks:
            if self.is_stateful and state is not None:
                substate = state.substate(component)
                inp, h, substate = component(h, state=substate, key=key)
                new_state = new_state.update(substate)
            else:
                inp, h = component(h, key=key)

        raw_sigma = self.to_sigma(h, key=key)

        if self.condition is not None:
            h_view = self.conditional_pos_enc(view_angle)
            h = jnp.concatenate([h, h_view], axis=-1)
            if self.is_stateful and state is not None:
                substate = state.substate(self.condition)
                h = self.condition(h, state=substate, key=key)
                new_state = new_state.update(substate)
            else:
                h = self.condition(h, key=key)

        raw_rgb = self.to_rgb(h, key=key)

        if self.is_stateful and state is not None:
            return raw_rgb, raw_sigma, new_state
        return raw_rgb, raw_sigma



class NeRF(INRModule):
    coarse_model: NeRFComponent
    fine_model: NeRFComponent
    num_coarse_samples: int
    num_fine_samples: int
    use_viewdirs: bool
    near: float
    far: float
    noise_std: float
    white_bkgd: bool
    lindisp: bool

    @property
    def is_stateful(self):
        return self.coarse_model.is_stateful or self.fine_model.is_stateful


    @classmethod
    def from_config(cls,
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
        key,
        initialization_scheme,
        initialization_scheme_kwargs,
        positional_encoding_layer,
        num_splits,
        post_processor,
        num_coarse_samples,
        num_fine_samples,
        use_viewdirs,
        near,
        far,
        noise_std,
        white_bkgd,
        lindisp):
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
                key,
                initialization_scheme,
                initialization_scheme_kwargs,
                positional_encoding_layer,
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
                        key,
                        initialization_scheme,
                        initialization_scheme_kwargs,
                        positional_encoding_layer,
                        num_splits,
                        post_processor
                )
        return cls(
            coarse_model=coarse_model,
            fine_model=fine_model,
            num_coarse_samples=num_coarse_samples,
            num_fine_samples=num_fine_samples,
            use_viewdirs=use_viewdirs,
            near=near,
            far=far,
            noise_std=noise_std,
            white_bkgd=white_bkgd,
            lindisp=lindisp

        )


    def __call__(self, ray_origins, ray_directions, randomized, state: Optional[eqx.nn.State] = None, *, key: jax.Array = jnp.array([0, 0])):
        """
        Forward pass through the NeRF model.
        
        Args:
            ray_origins: jnp.ndarray(float32), shape [batch_size, 3], ray origins
            ray_directions: jnp.ndarray(float32), shape [batch_size, 3], ray directions
            randomized: bool, whether to use randomized sampling
            state: Optional state for stateful layers
            key: Random number generator key
        
        Returns:
            If not stateful:
                List of tuples [(coarse_rgb, coarse_disp, coarse_acc), (fine_rgb, fine_disp, fine_acc)]
            If stateful:
                Same list of tuples plus updated state
        """
        key_gen = key_generator(key)
    
        # Normalize ray directions
        viewdirs = ray_directions / jnp.linalg.norm(ray_directions, axis=-1, keepdims=True)
    
        # Sample along rays
        z_vals, samples = self.sample_along_ray(
            next(key_gen),
            ray_origins,  # [batch_size, 3]
            ray_directions,  # [batch_size, 3] 
            self.num_coarse_samples,
            self.near,
            self.far,
            randomized,
            self.lindisp,
        )

        new_state = state

        # Run coarse model
        if self.use_viewdirs:
            if self.coarse_model.is_stateful and state is not None:
                substate = state.substate(self.coarse_model)
                raw_rgb, raw_sigma, substate = self.coarse_model(samples, viewdirs, state=substate)
                new_state = new_state.update(substate)
            else:
                raw_rgb, raw_sigma = self.coarse_model(samples, viewdirs)
        else:
            if self.coarse_model.is_stateful and state is not None:
                substate = state.substate(self.coarse_model)
                raw_rgb, raw_sigma, substate = self.coarse_model(samples, state=substate)
                new_state = new_state.update(substate)
            else:
                raw_rgb, raw_sigma = self.coarse_model(samples)

        # Add noise to regularize the density predictions if needed
        raw_sigma = self.add_gaussian_noise(
            next(key_gen),
            raw_sigma,
            self.noise_std,
            randomized,
        )
        rgb = jax.nn.sigmoid(raw_rgb)
        sigma = jax.nn.relu(raw_sigma)

        # Volumetric rendering
        comp_rgb, disp, acc, weights = self.volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            ray_directions,
            white_bkgd=self.white_bkgd,
        )
        ret = [
            (comp_rgb, disp, acc),
        ]

        # Hierarchical sampling based on coarse predictions
        if self.num_fine_samples > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_vals, samples = self.sample_pdf(
                next(key_gen),
                z_vals_mid,
                weights[..., 1:-1],
                ray_origins,
                ray_directions,
                z_vals,
                self.num_fine_samples,
                randomized,
            )

            # Run fine model
            if self.use_viewdirs:
                if self.fine_model.is_stateful() and state is not None:
                    substate = state.substate(self.fine_model)
                    raw_rgb, raw_sigma, substate = self.fine_model(samples, viewdirs, state=substate)
                    new_state = new_state.update(substate)
                else:
                    raw_rgb, raw_sigma = self.fine_model(samples, viewdirs)
            else:
                if self.fine_model.is_stateful() and state is not None:
                    substate = state.substate(self.fine_model)
                    raw_rgb, raw_sigma, substate = self.fine_model(samples, state=substate)
                    new_state = new_state.update(substate)
                else:
                    raw_rgb, raw_sigma = self.fine_model(samples)

            raw_sigma = self.add_gaussian_noise(
                next(key_gen),
                raw_sigma,
                self.noise_std,
                randomized,
            )
            rgb = jax.nn.sigmoid(raw_rgb)
            sigma = jax.nn.relu(raw_sigma)
            comp_rgb, disp, acc, unused_weights = self.volumetric_rendering(
                rgb,
                sigma,
                z_vals,
                ray_directions,
                white_bkgd=self.white_bkgd,
            )
            ret.append((comp_rgb, disp, acc))

        if self.coarse_model.is_stateful() or self.fine_model.is_stateful():
            return ret, new_state
        return ret

    @staticmethod
    def cast_ray(z_vals, origin, direction):
        """
        Cast ray through pixel positions.
        """
        # return origin[None, :] + z_vals[:, None] * direction[ None, :]
        origin = origin[None, :, :]  # [1, batch_size, 3]
        direction = direction[None, :, :]  # [1, batch_size, 3]

        # Add batch and xyz dimensions to z_vals
        z_vals = z_vals[:, None, None]  # [num_samples, 1, 1]

        return origin + z_vals * direction


    def sample_along_ray(self, key, origin, direction, num_samples, near, far, randomized, lindisp):
        """
        Sample along a ray.

        :param key: jnp.ndarray(float32), [2,], random number generator.
        :param origin: jnp.ndarray(float32), [3], ray origin.
        :param direction: jnp.ndarray(float32), [3], ray direction.
        :param num_samples: int, the number of samples.
        :param near: float, the distance to the near plane.
        :param far: float, the distance to the far plane.
        :param randomized: bool, use randomized samples.
        :param lindisp: bool, sample linearly in disparity rather than in depth.

        :return: z_vals: jnp.ndarray(float32), [num_samples].
        :return: coords: jnp.ndarray(float32), [num_samples, 3].


        """

        t_vals = jnp.linspace(0., 1., num_samples)
        if lindisp:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
        else:
            z_vals = near * (1. - t_vals) + far * t_vals

        if randomized:
            mids = .5 * (z_vals[1:] + z_vals[:-1])
            upper = jnp.concatenate([mids, z_vals[-1:]], -1)
            lower = jnp.concatenate([z_vals[:1], mids], -1)
            t_rand = random.uniform(key, [num_samples])
            z_vals = lower + (upper - lower) * t_rand

        coords = self.cast_ray(z_vals, origin, direction)

        return z_vals, coords

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
    def volumetric_rendering(rgb, sigma, z_vals, direction, white_bkgd):
        """Volumetric Rendering Function.

        :param rgb: jnp.ndarray(float32), color, [num_samples, 3].
        :param sigma: jnp.ndarray(float32), density, [num_samples, 1].
        :param z_vals: jnp.ndarray(float32), [num_samples].
        :param direction: jnp.ndarray(float32), [3].
        :param white_bkgd: bool.

        :return: comp_rgb: jnp.ndarray(float32), [3].
        """
        eps = 1e-10
        dists = jnp.concatenate([
            z_vals[1:] - z_vals[:-1],
            jnp.broadcast_to(1e10, z_vals[:1].shape)
        ], -1)
        dists = dists * jnp.linalg.norm(direction[None, :], axis=-1)
        # Note that we're quietly turning sigma from [..., 0] to [...].
        alpha = 1.0 - jnp.exp(-sigma[:, 0] * dists)
        accum_prod = jnp.concatenate([
            jnp.ones_like(alpha[:1], alpha.dtype),
            jnp.cumprod(1.0 - alpha[:-1] + eps, axis=-1)
        ],
            axis=-1)
        weights = alpha * accum_prod

        comp_rgb = (weights[:, None] * rgb).sum(axis=-2)
        depth = (weights * z_vals).sum(axis=-1)
        acc = weights.sum(axis=-1)
        # Equivalent to (but slightly more efficient and stable than):
        #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
        inv_eps = 1 / eps
        disp = acc / depth
        disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
        if white_bkgd:
            comp_rgb = comp_rgb + (1. - acc[:, None])
        return comp_rgb, disp, acc, weights


    @staticmethod
    def piecewise_constant_pdf(key, bins, weights, num_samples, randomized):
        """Piecewise-Constant PDF sampling.

        :param key: jnp.ndarray(float32), [2,], random number generator.
        :param bins: jnp.ndarray(float32), [num_bins + 1].
        :param weights: jnp.ndarray(float32), [num_bins].
        :param num_samples: int, the number of samples.
        :param randomized: bool, use randomized samples.

        :return: z_samples: jnp.ndarray(float32), [num_samples].
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
        cdf = jnp.minimum(1, jnp.cumsum(pdf[:-1], axis=-1))
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
        mask = u[:, None] >= cdf[:, None]

        def find_interval(x):
            # Grab the value where `mask` switches from True to False, and vice versa.
            # This approach takes advantage of the fact that `x` is sorted.
            x0 = jnp.max(jnp.where(mask, x[:, None], x[:1, None]), -2)
            x1 = jnp.min(jnp.where(~mask, x[:, None], x[-1:, None]), -2)
            return x0, x1

        bins_g0, bins_g1 = find_interval(bins)
        cdf_g0, cdf_g1 = find_interval(cdf)

        t = jnp.clip(jnp.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        samples = bins_g0 + t * (bins_g1 - bins_g0)

        # Prevent gradient from backprop-ing through `samples`.
        return lax.stop_gradient(samples)

    def sample_pdf(self, key, bins, weights, origin, direction, z_vals, num_samples, randomized):
        """Hierarchical sampling.

        Args:
          key: jnp.ndarray(float32), [2,], random number generator.
          bins: jnp.ndarray(float32), [num_bins + 1].
          weights: jnp.ndarray(float32), [num_bins].
          origin: jnp.ndarray(float32), [3], ray origin.
          direction: jnp.ndarray(float32), [3], ray direction.
          z_vals: jnp.ndarray(float32), [num_coarse_samples].
          num_samples: int, the number of samples.
          randomized: bool, use randomized samples.

        Returns:
          z_vals: jnp.ndarray(float32), [num_coarse_samples + num_fine_samples].
          points: jnp.ndarray(float32), [num_coarse_samples + num_fine_samples, 3].
        """
        z_samples = self.piecewise_constant_pdf(key, bins, weights, num_samples, randomized)
        # Compute united z_vals and sample points
        z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
        coords = self.cast_ray(z_vals, origin, direction)
        return z_vals, coords
