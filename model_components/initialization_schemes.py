""" 
Module for intialization schemes
Initializaiton schemes should be callables with the following signature:

:param in_size: integer indicating the input size of the layer that is to be created
:param out_size: integer indicating the output size of said layer
... any parameters that are required activation kwargs for tha layer that are relevant to the initializaiton scheme
:param num_splits: integer indicating how many weights matrices and bias vectors should be created for the layer 
:keyword only param layer_type: type[INRLayer] indicating what layer type to create an instance of
:keyword only param key: prng key
:keyword only param is_first_layer: bool indicating whether this layer will serve as the first layer in the network
:var keywrod argument additional_layer_kwargs: additional kwargs to be passed to the INRLayer. 

:return: an instance of layer_type. 
"""
from typing import Optional
import jax
from jax import numpy as jnp
from model_components.inr_layers import INRLayer

def siren_scheme(in_size:int, out_size:int,  w0:float,  num_splits=1,*, layer_type:type[INRLayer], key:jax.Array, is_first_layer:bool,  **additional_layer_kwargs):
    """the initialization scheme from the SIREN paper, but able to initialize any layer type that has the same activation kwargs
    
    :param in_size: size of the input to the layer
    :param out_size: size of the output of the layer
    :param w0: value of the w0 hyper parameter from the SIREN paper    
    :param num_splits: is ignored (TODO implement using this)
    :param layer_type: a subclass of INRLayer that should be initialized with weights and biases from this initializaiton scheme (keyword only)
    :param key: key for random number generator (keyword only)
    :param is_first_layer: whether this is the first layer in an INR or not (keyword only)
    :param additional_layer_kwargs: optional additional layer_kwargs necessary for the initialization of layer_type.
    

    :raises: ValueError if any other activation_kwargs than 'w0' are provided

    :return: a SirenLayer with weights and biases initialized according to the scheme provided in the original SIREN paper
    """
    cls = layer_type
    activation_kwargs = dict(additional_layer_kwargs) if additional_layer_kwargs is not None else {}
    activation_kwargs["w0"] = w0
    activation_kwargs = cls._check_keys(activation_kwargs)
    
    w_key, b_key = jax.random.split(key)

    if is_first_layer:
        lim = 1./in_size# from https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L630
    else:
        lim = jnp.sqrt(6./in_size)/w0  # from https://arxiv.org/pdf/2006.09661.pdf subsection.3.2 and appendix 1.5 and https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L627
    
    weight = jax.random.uniform(
        key=w_key,
        shape=(out_size, in_size),
        minval=-lim, 
        maxval=lim
        )
        
    bias = jax.random.uniform(
        key=b_key,
        shape=(out_size,),
        minval=-1,
        maxval=1
    )
    bias_factor = jnp.pi/jnp.sqrt(jnp.sum(jnp.square(weight), axis=1)) # from https://arxiv.org/pdf/2102.02611.pdf page 6 third paragaph
    bias = bias_factor * bias

    return cls(weight, bias, **activation_kwargs)