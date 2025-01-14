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




def finer_scheme(in_size: int, out_size: int, w0: float, bias_k:float, num_splits=1, *, layer_type: type[INRLayer], key: jax.Array, is_first_layer: bool, **additional_layer_kwargs):
    """
    Initialization scheme for FINER layers using variable-periodic activation functions.

    :param in_size: Size of the input to the layer.
    :param out_size: Size of the output of the layer.
    :param omega: Frequency scaling parameter for variable-periodic activations.
    :param bias_k: Range for bias initialization; biases are sampled from U(-k, k).
    :param num_splits: Number of weight matrices and bias vectors (ignored here).
    :param layer_type: Type of INRLayer to initialize.
    :param key: PRNG key for randomness.
    :param is_first_layer: Indicates whether this is the first layer in the network.
    :param additional_layer_kwargs: Additional arguments for the INRLayer.

    :return: An instance of layer_type initialized with FINER++ scheme.
    """
    cls = layer_type
    activation_kwargs = {'w0':w0}
    
    activation_kwargs = cls._check_keys(activation_kwargs)
    
    k = bias_k

    
    w_key, b_key = jax.random.split(key)

    # Weight initialization using SIREN-like scheme, adapted for FINER++
    if is_first_layer:
        weight_lim = 1.0 / in_size
    else:
        weight_lim = jnp.sqrt(6.0 / in_size) / w0

    weights = jax.random.uniform(
        key=w_key,
        shape=(out_size, in_size),
        minval=-weight_lim,
        maxval=weight_lim
    )

    # Bias initialization with a larger range to support variable periodicity
    biases = jax.random.uniform(
        key=b_key,
        shape=(out_size,),
        minval=-k,
        maxval=k
    )

    return cls(weights, biases, **activation_kwargs)



def scale_initialization_scheme(init_scheme: Callable, scale_factor: float, *args, **kwargs) -> Callable:
    """Creates a scaled version of an initialization scheme.
    
    :param init_scheme: The initialization scheme to scale
    :param scale_factor: Factor to scale the weights and biases by
    :param args: Positional arguments to pass to the initialization scheme
    :param kwargs: Keyword arguments to pass to the initialization scheme
    :return: A new initialization scheme that scales parameters based on the scheme type
    """
    # Define which parameters should be scaled for each initialization scheme
    SCALABLE_PARAMS = {
        "siren_scheme": ["w0"],
        "finer_scheme": ["w0", "bias_k"],
        "gaussian_scheme": ["inverse_scale"],
        # Add new schemes and their scalable parameters here
    }

    # Get initialization scheme name from the function
    scheme_name = init_scheme.__name__

    # Get list of parameters that should be scaled for this scheme
    scalable_params = SCALABLE_PARAMS.get(scheme_name, [])

    # Scale only the designated parameters if they exist in kwargs
    for param in scalable_params:
        if param in kwargs:
            kwargs[param] *= scale_factor

    # Return initialization with scaled parameters
    return init_scheme(*args, **kwargs)

# # Scale parameters based on initialization scheme type
#     if scheme_name == "siren_scheme":
#         if 'w0' in kwargs:
#             kwargs['w0'] *= scale_factor
#     elif scheme_name == "finer_scheme":
#         if 'w0' in kwargs:
#             kwargs['w0'] *= scale_factor
#         if 'bias_k' in kwargs:
#             kwargs['bias_k'] *= scale_factor
#     elif scheme_name == "gaussian_scheme":
#         if 'inverse_scale' in kwargs:
#             kwargs['inverse_scale'] *= scale_factor
