""" 
Classes meant to serve as components of (equivariant) hypernetworks.

The actual hypernetwork can be created as an EquivariantHyperNetwork instance.
This class takes a graph network and an equivariant inr creator as arguments.
    One example of a graph network is the PAINNNetwork class in this module.
        The abstract base class for a graph network is the GraphNetwork class in this module.
    One example of an equivariant inr creator is the TwoWayINRCreator class in this module.
        The abstract base class for an equivariant inr creator is the EquivariantINRCreator class in this module.

Moreover, this module contains classes that can be used as components of the equivariant inr creator:
    HiddenINRLayerAssemblyLayer: a class for creating hidden layers of an INR
    InputEquivariantINRLayerAssemblyLayer: a class for creating input layers of an INR
    OutputINRLayerAssemblyLayer: a class for creating output layers of an INR
These classes make use of the assumption behind INRLayers from model_components.inr_layers that the __init__ method
of these classes takes the weights, biases, and activation_kwargs of the intended layer as input.

In this module, equivariance is typically meant to be with respect to orthogonal transformations and possibly translations.
"""
from typing import Union, Optional

import jax
from jax import numpy as jnp
import equinox as eqx

from model_components.inr_layers import INRLayer, Linear


class HiddenINRLayerAssemblyLayer(eqx.Module):
    # based on https://arxiv.org/pdf/2011.12026v2.pdf

    shared_matrix: Union[jax.Array, list[jax.Array]]
    low_rank: int
    in_size: int
    out_size: int
    num_splits: int
    layer_type: type[INRLayer] 
    layer_kwargs: dict = eqx.field(static=True)
    _total_in: int
    _total_out: int
    has_bias: bool
    input_size: int
    _trainable_inverse_matrix_scale: jax.Array
    _inverse_matrix_scale_lower_bound: float
    _multi_call: bool

    def __init__(
            self, 
            in_size:int, 
            out_size:int, 
            low_rank:int, 
            layer_type:type[INRLayer], 
            layer_kwargs:dict,
            key: jax.Array, 
            initial_inv_scale: float, 
            minimal_inv_scale: float,
            num_splits:int = 1,
            has_bias:bool = True, 
            ):
        """
        Layer for turning a vector of parameters into a *hidden* INR layer.
        The idea is based on the "Factorized Multiplicative Modulation" from 
        the paper "Adversarial Generation of Continuous Images" by Skorokhodov et al.
        (https://arxiv.org/pdf/2011.12026v2.pdf Section 3.2)

        The weights of the INR layer are the Hadamard product of a shared matrix and a low-rank modulating matrix.
        This modulating matrix is parameterized by the input to the HiddenINRLayerAssemblyLayer.
        In order to make the distribution of the output weights at the start of training similar to
        the initialization distribution for the corresponding INR layer type,
        we make use of a (scalar) trainable inverse matrix scale.

        :parameter in_size: the size of the input to the INR layer
        :parameter out_size: the size of the output of the INR layer
        :parameter low_rank: the rank of the modulating matrix
        :parameter layer_type: the type of INR layer to be created
        :parameter layer_kwargs: activation_kwargs for the INR layer
        :parameter key: a jax random key for initializing the shared matrix
        :parameter initial_inv_scale: the initial value of the trainable part of the inverse matrix scale
            NB the resulting initial value of the inverse matrix scale will be initial_inv_scale + minimal_inv_scale
        :parameter minimal_inv_scale: the minimal value of the inverse matrix scale (not trainable)
        :parameter num_splits: the number of splits used for the INR layer
        :parameter has_bias: whether the INR layer has a bias
            NB biasless INR layers are currently not implemented
        
        """
        if not has_bias:
            raise NotImplementedError("INRLayers without biases have currently not been implemented")

        self.in_size = in_size
        self.out_size = out_size
        self.low_rank = low_rank
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        is_first_layer = False

        self._total_in = in_size * low_rank  # number inputs required per in-matrix
        self._total_out = out_size * low_rank # number of inputs required per out-matrix
        self.has_bias = has_bias

        self._trainable_inverse_matrix_scale = jnp.array([initial_inv_scale])#jnp.array([7.9])
        self._inverse_matrix_scale_lower_bound = jnp.array([minimal_inv_scale])
        # at the start of training, the distribution of the full matrix should be close
        # to the initialization scheme for the relevant inr layer
        # this scale helps achieve that


        # to get the shared matrix, initialize a corresponding INR layer and use its weights
        layer_for_initial_weights = self.layer_type.from_config(
            in_size=in_size,
            out_size=out_size,
            num_splits=num_splits,
            key=key,
            is_first_layer=is_first_layer,
            **layer_kwargs
        )
        weights = layer_for_initial_weights.weights
        if isinstance(weights, (tuple, list)):
            self.num_splits = len(weights)  # just in case the layer modifies this
            self.shared_matrix = list(weights)
            self._multi_call = True
        else:
            self.num_splits = 1
            self.shared_matrix = weights
            self._multi_call = False 

        if self.has_bias:
            self.input_size = self.num_splits*(self._total_in + self._total_out + self.out_size)
        else: 
            self.input_size = self.num_splits*(self._total_in + self._total_out)
    
    def __call__(self, h:jax.Array)->INRLayer:
        # NB some INR layers may use multiple (weights, bias) pairs.
        # for those, we use self._multi_matrix_call
        # for INR layers with a single weights matrix and bias vector,
        # we use self._single_matrix_call
        if self._multi_call:
            return self._multi_matrix_call(h)
        return self._single_matrix_call(h)

    @property
    def inverse_matrix_scale(self):
        return jnp.maximum(self._trainable_inverse_matrix_scale, 0.) + jax.lax.stop_gradient(self._inverse_matrix_scale_lower_bound)

    def prepare_single_matrix_and_bias(self, h:jax.Array, shared_matrix:jax.Array)->list[jax.Array]:
        """
        Using a vector of input features and a shared matrix, 
        construct the a weight matrix and bias vector for an INR of type self.layer_type

        :parameter h: input features of shape (self._total_in + self._total_out + self.out_size,)
            (or of shape (self._total_in + self._total_out,) if the inr is not supposed to have a bias)
        :parameter shared_matrix: shared matrix with shape (self.out_size, self.in_size)
        :return: [weights, bias] (or just [weights] if the inr layer is not supposed to have a bias).
        """
        h_in = h[:self._total_in]
        h_out = h[self._total_in:self._total_in + self._total_out]

        in_matrix = jnp.reshape(h_in, (self.low_rank, self.in_size))
        out_matrix = jnp.reshape(h_out, (self.out_size, self.low_rank))

        full_matrix = 2 * shared_matrix * jax.nn.sigmoid(out_matrix @ in_matrix / self.inverse_matrix_scale)  # TODO test if the 2* works well

        args = [full_matrix]
        if self.has_bias:
            args.append(h[self._total_in + self._total_out:])
        return args

    def _single_matrix_call(self, h:jax.Array)->INRLayer:
        """
        Forward pass in case the inr layer is to have only a single weights matrix and bias vector
        """
        args = self.prepare_single_matrix_and_bias(h, self.shared_matrix)
        return self.layer_type(*args, **self.layer_kwargs)
    
    def _multi_matrix_call(self, h:jax.Array)->INRLayer:
        """
        Forward pass in case the inr layer is to have multiple (weights, bias) pairs.
        """
        multi_h = jnp.split(h, self.num_splits)
        args = zip(*[
            self.prepare_single_matrix_and_bias(inputs, shared_matrix)
            for inputs, shared_matrix in zip(multi_h, self.shared_matrix)
            ])  # the zip transposes this structure from [(weight, bias), ...] to ((weight, ...), (bias, ...))
        # notably, if there are no biases, this should still work
        return self.layer_type(*args, **self.layer_kwargs)


class FullINRLayerAssemblyLayer(eqx.Module):
    """
    Layer for turning a vector of parameters into the input and output layers of an INR.
    The input to this layer should be a single vector of parameters.
    The output of this layer should be a single Linear INR layer.

    :parameter in_size: the size of the input to the INR layer
    :parameter out_size: the size of the output of the INR layer
    :parameter layer_type: the type of layer to be created 
        default: Linear
    :parameter layer_kwargs: an optional dict of activation_kwargs to be passed to layer_type
        default: None, which results in an empty dict
    """
    in_size: int
    out_size: int

    layer_type:type[INRLayer] = Linear
    layer_kwargs:Optional[dict] = eqx.field(static=True, default=None)


    def __call__(self, h:jax.Array)->Linear:
        """
        :parameter h: input features of shape (in_size*out_size + out_size,)
        :return: Linear INR layer
        """
        weights, bias = h[:-self.out_size], h[-self.out_size:]
        weights = jnp.reshape(weights, (self.out_size, self.in_size))
        layer_kwargs = self.layer_kwargs or {}
        return self.layer_type(weights, bias, **layer_kwargs)
    
    @property
    def input_size(self):
        return self.in_size*self.out_size + self.out_size
    

