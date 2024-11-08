""" 
INRLayer classes for Implicit Neural Representations / Coordinate-MLPs / Neural Fields
The main difference with normal MLP layers is that the __init__ method of these layers
takes the weights, biases, and activation_kwargs as input, so as to make it easier to use them with hyper networks.
To initialize a layer using their typical initialization scheme based on hyperparameters such as input and output size,
one should use the from_config classmethod.
Alternatively one can write an initialization function that comes up with the weights and biases using a specific initialization scheme,
and then passes those to the __init__ function of the required class.
"""
import abc
from typing import Union, Callable, Optional
from collections.abc import Sequence
import inspect

import jax
from jax import numpy as jnp
import equinox as eqx

from common_jax_utils import key_generator
import model_components.auxiliary as aux
import model_components.activation_functions as act

class INRLayer(eqx.Module):
    """
    INRLayer (abstract) base class for INR layers
    To create a new INRLayer type implement three things:
    1) define the _activation_function
    2) provide an initialization scheme for the weights and biases by defining a from_config classmethod.
    3) provide a (frozen) set of allowed_keys

    Additionally one should provide a frozen set, allowed_keys, of allowed key word arguments for the activation function
    and provide a boolean, allows_multiple_weights_and_biases, indicating whether the layer can have a list/tuple of weight matrices and bias vectors, or allows only one of each.

    creation of instances of INRLayer subclasses can happen in two ways:
    1) through __init__ by providing the values for weights, biases, and activation_kwargs
    2) through from_config or complex_from_config by providing the hyperparameters needed for the creation of the weights and biases including the activation_kwargs
    """
    weights: Union[jax.Array, list[jax.Array], tuple[jax.Array]]
    biases: Union[jax.Array, list[jax.Array], tuple[jax.Array]]

    activation_kwargs: dict = eqx.field(static=True)  # think w0 for siren or inverse_scale for gaussian
    _activation_function: eqx.AbstractClassVar[Callable]
    allowed_keys: eqx.AbstractClassVar[frozenset[Union[str, tuple[str, aux.ANNOT]]]]  # the keys that should be present in activation_kwargs
    allows_multiple_weights_and_biases: eqx.AbstractClassVar[bool]

    @classmethod
    def _check_keys(cls, activation_kwargs):
        """
        check that all keys specified in cls.allowed_keys are present in activation_kwargs
            if not, raise a ValueError
        return a filtered version of activation_kwargs where only the keys in cls.allowed_keys are present

        This filtering makes it easer to perform wandb sweeps where we vary the type of INR layer
        """
        return aux.filter_allowed_keys_or_raise_value_error(activation_kwargs, *cls.allowed_keys) 
        
    @classmethod
    def _check_weights_and_biases(cls, weights, biases):
        """ 
        Check if the types of weights and biases agree with cls.allows_multiple_weights_and_biases
        and whether, if weights and biases are tuples/lists, they are of equal length.
        """
        w_seq = isinstance(weights, (tuple, list))
        b_seq = isinstance(biases, (tuple, list))

        if not cls.allows_multiple_weights_and_biases:
            if w_seq or b_seq:
                raise ValueError(f"{cls.__name__} does not allow for multiple weights or biases. Got {type(weights)=} and {type(biases)=}.")
            return
        #else
        if w_seq ^ b_seq:  # xor
            raise ValueError(f"weights and biases should either both be a tuple/list of jax.Array objects, or both be a jax.Array. Got {type(weights)=} but {type(biases)=}.")
        if w_seq:
            w_len = len(weights)
            b_len = len(biases)
            if w_len!=b_len:
                raise ValueError(f"When providing sequences of weights and biases, the sequences should be of equal length (got len(weights)={w_len} but len(biases)={b_len})")

    def activation_function(self, *args):
        """ 
        Apply the activation function to the input using the kwargs stored in self.activation_kwargs
        """
        return self._activation_function(*args, **self.activation_kwargs)
    
    def __init__(self, weights, biases, **activation_kwargs):
        """ 
        Initialise an INRLayer from its weights, biases, and activation_kwargs
        """
        self._check_weights_and_biases(weights, biases)
        self.weights = weights
        self.biases = biases
        activation_kwargs = self._check_keys(activation_kwargs)
        self.activation_kwargs = activation_kwargs

    def __init_subclass__(cls):
        """__init_subclass__ 
        Modify the signature of cls.from_config based on cls.allowed_keys
        To enable the use of the tools from common_dl_utils, the signature of from_config is made to specify exactly what parameters need to be provided.
        """
        from_config_signature = inspect.signature(cls.from_config)
        # we want to replace the **activation_kwargs with the parameters in cls.allowed_keys
        # we need to add cls to the parameters so that we don't lose the first parameter in the signature
        # upon calling classmethod
        parameters = {'cls': inspect.Parameter('cls', inspect.Parameter.POSITIONAL_OR_KEYWORD)}
        parameters.update(from_config_signature.parameters)
        #print(f"{cls.__name__}: {parameters=}")  # this was for debugging
        # TODO find out why an _InitableModule seems to be created only when using from_config and not using normal initialization
        # or at leas why _InitableModule calls this __init_subclass__ only in the former and not in the latter case
        # (not that it really matters)
        parameters.pop('activation_kwargs', None)
        for key_annot in cls.allowed_keys:
            if isinstance(key_annot, str):
                parameters[key_annot] = inspect.Parameter(
                    name=key_annot,
                    kind=inspect.Parameter.KEYWORD_ONLY
                )
            else:
                param_name, annotation = key_annot
                parameters[param_name] = inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=annotation
                )
        new_signature = from_config_signature.replace(parameters=parameters.values())
        
        # https://docs.python.org/3/library/stdtypes.html#methods
        # tells us we can't set attributes on (bound) methods
        # but need to set them on method.__func__ instead.
        cls.from_config.__func__.__signature__ = new_signature
        # TODO maybe consider also changing the signature of __init__ in the same way

    
    @classmethod
    @abc.abstractmethod
    def from_config(cls, in_size:int, out_size:int, num_splits:int=1, *, key:jax.Array, is_first_layer:bool, **activation_kwargs):
        """
        Abstract classmethod
        This should initialize the weights and biases based on hyperparameters and a prng key.

        :parameter in_size: dimensionality of the input to the layer
        :parameter out_size: dimensionality of the output of the layer
        :parameter num_splits: some layers may process the same input using multiple weights matrices and bias vectors. In those layers,
            num_splits determines the number of weights matrices and bias vectors that is to be used.
        :parameter key: prng key for initializing with random weights and biases
        :parameter is_first_layer: some layers may use a different initialization scheme when they are the first layer in an INR. This boolean
            parameter is used to indicate to such layers whether they are the first layer in an INR or not.
        :parameter activation_kwargs: any additional hyperparameters necessary for the layer. E.g. w0 for SIREN
        """
        pass

    def __call__(self, x:jax.Array, *, key:Optional[jax.Array]):
        # key just to have it be compatible with eqx.nn.Sequential
        if isinstance(self.weights, (list, tuple)):
            # when num_splits > 1
            wxb = [w@x + b for w, b in zip(self.weights, self.biases)]
        else:
            # when num_splits=1
            wxb = (self.weights@x + self.biases,)
        return self.activation_function(*wxb)
    
    @classmethod
    def complex_from_config(cls, in_size, out_size, num_splits=1, *, key, is_first_layer, **activation_kwargs):
        """
        Like from_config, but creates a layer with complex weights and biases.
        """
        key_1, key_2 = jax.random.split(key)
        real_part = cls.from_config(in_size=in_size, out_size=out_size, num_splits=num_splits, key=key_1, is_first_layer=is_first_layer, **activation_kwargs)
        imaginary_part = cls.from_config(in_size=in_size, out_size=out_size, num_splits=num_splits, key=key_2, is_first_layer=is_first_layer, **activation_kwargs)
        weights = jax.tree_map(jax.lax.complex, real_part.weights, imaginary_part.weights)
        biases = jax.tree_map(jax.lax.complex, real_part.biases, imaginary_part.biases)
        return cls(weights, biases, **activation_kwargs)

class SirenLayer(INRLayer):
    """
    Siren layer from "Implicit Neural Representations with Periodic Activation functions" by Sitzmann et al. 
    :param weights: jax.Array containing the weights of the linear part
    :param biases: jax.Array containing the bias of the linear part
    :param w0: w0 hyperparameter as introduced in the SIREN paper by Sitzmann et al.

    No other activation_kwargs than w0 are allowed
    """
    allowed_keys = frozenset({'w0'})
    allows_multiple_weights_and_biases = False

    @classmethod
    def from_config(cls, in_size:int, out_size:int, num_splits:int=1, *, key:jax.Array, is_first_layer:bool, **activation_kwargs):
        """from_config create a layer from hyperparameters

        :param in_size: size of the input
        :param out_size: size of the output
        :param num_splits: ignored, defaults to 1
        :param key: key for random number generator (keyword only)
        :param is_first_layer: whether this is the first layer in an INR or not (keyword only)
        :param w0: value of the w0 hyper parameter from the SIREN paper (keyword only)

        :raises: ValueError if 'w0' is not provided

        :return: a SirenLayer with weights and biases initialized according to the scheme provided in the original SIREN paper
        """
        activation_kwargs = cls._check_keys(activation_kwargs)
        w0 = activation_kwargs['w0']
        
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

    @staticmethod
    def _activation_function(x, w0):
        return jnp.sin(w0*x)


class RealWire(SirenLayer):
    """ 
    Example implementation of the Real WIRE layer (without complex numbers) from https://arxiv.org/abs/2301.05187 (Section 3.4 on "Alternative forms of WIRE")
    :parameter weights: `jax.Array` containing the weights matrix or sequence of `jax.Array`s containint the various weights matrices for the various splits (in WIRE 2D or 3D)
        NB if a sequence of weights matrices is provided, their shape should be identical.
    :parameter biases: `jax.Array` containing the bias vector or sequence of `jax.Array`s containing the various bias vectors for the various splits (in WIRE 2D or 3D)
        NB if a sequence of bias vectors is provided, the length of the sequence should be identical to the length of the sequence of weights matrices.
    :parameter w0: frequency parameter of the Gabor Wavelet (\omega_0 in the paper)
    :parameter s0: inverse scale or width parameter of the Gaussian (s_0 in the paper)

    The initialization scheme is that of SIREN
    """
    allowed_keys = frozenset({'w0', 's0'})
    allows_multiple_weights_and_biases = True

    @staticmethod
    def _activation_function(x, w0, s0):
        return act.real_wire(x, s0=s0, w0=w0)
    
    # TODO write new from_config function that allows for WIRE 2D/3D etc.
    
    @staticmethod
    def _initialize_single_weights_and_bias(in_size:int, out_size:int, w0:float, is_first_layer:bool, key:jax.Array)->tuple[jax.Array, jax.Array]:
        """ 
        Initialize a single weights matrix and bias vector using the initialization scheme for SIREN
        :parameter in_size: dimensionality of the input to the layer
        :parameter out_size: dimensionality of the output of the layer
        :parameter w0: frequency parameter (\omega_0 in both the SIREN and the WIRE paper)
        """
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
        return weight, bias
    
    @classmethod
    def from_config(cls, in_size:int, out_size:int, num_splits:int=1, *, key:jax.Array, is_first_layer:bool, **activation_kwargs):
        """from_config create a layer from hyperparameters

        :parameter in_size: size of the input
        :parameter out_size: size of the output
        :parameter num_splits: number of weights matrices used, defaults to 1
            Set this to 2 for WIRE2D and 3 for WIRE3D (etc.)
        :parameter key: key for random number generator (keyword only)
        :parameter is_first_layer: whether this is the first layer in an INR or not (keyword only)
        :parameter w0: frequency parameter (\omega_0 in both the SIREN and the WIRE paper)
        :parameter s0: inverse scale or width parameter of the Gaussian (s_0 in the paper)

        :raises: ValueError if 'w0' or 's0' is not provided

        :return: a RealWIRE with weights and biases initialized according to the scheme provided in the original SIREN paper
        """
        activation_kwargs = cls._check_keys(activation_kwargs)
        w0 = activation_kwargs['w0']
        s0 = activation_kwargs['s0']

        if num_splits == 1:
            weight, bias = cls._initialize_single_weights_and_bias(in_size=in_size, out_size=out_size, w0=w0, is_first_layer=is_first_layer, key=key)
            return cls(weight, bias, w0=w0, s0=s0)
        
        keys = jax.random.split(key, num_splits)
        weights, biases = jax.vmap(lambda k: cls._initialize_single_weights_and_bias(in_size=in_size, out_size=out_size, w0=w0, is_first_layer=is_first_layer, key=k))(keys)
        weights = jnp.unstack(weights)
        biases = jnp.unstack(biases)
        return cls(weights, biases, w0=w0, s0=s0)


class Linear(INRLayer):
    """
    Linear INRLayer
    :param weights: jax.Array containing the weights maxtrix
    :param biases: jax.Array containing the bias
    NB no activation_kwargs allowed
    """

    allowed_keys = frozenset()
    allows_multiple_weights_and_biases = False

    @classmethod
    def from_config(cls, in_size:int, out_size:int, num_splits:int=1, *, key:jax.Array, is_first_layer:bool, **activation_kwargs):
        """from_config create a layer from hyperparameters

        :param in_size: size of the input
        :param out_size: size of the output
        :param num_splits: ignored, defaults to 1
        :param key: key for random number generator (keyword only)
        :param is_first_layer: ignored (keyword only)

        :raises: ValueError if any activation_kwargs are provided

        :return: a Linear layer with weights and biases initialized according a uniform distribution with bounds +/- 1/sqrt(in_size)
        
        NB this INRLayer accepts no activationn_kwargs
        """
        activation_kwargs = cls._check_keys(activation_kwargs)
        w_key, b_key = jax.random.split(key)

        lim = 1./jnp.sqrt(in_size)
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

        return cls(weight, bias, **activation_kwargs)

    @staticmethod
    def _activation_function(x):
        return x

class GaussianINRLayer(INRLayer):
    """
    Gaussian INR layer introduced in the paper "Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs" by S. Ramasinghe and S. Lucey
    This type of INR is claimed by said authors to be less sensitive to initialization than SIREN
    :param weights: jax.Array containing the weights of the linear part
        Alternatively, this can be a list or tuple of jax.Arrays of the same size containing multiple weights matrices
        The bumps coming from different "paths" will be multiplied (summed in log-space)
    :param biases: jax.Array containing the bias of the linear part
        Alternatively, this can be a list or tuple of jax.Arrays of the same size containing multiple bias vectors
        This list or tuple should be the same length as the one for weights
    :param inverse_scale: float scale hyper parameter for $x\mapsto e^(-|inverse_scale\cdot x|^2)$
    
    NB no other activation_kwargs than inverse_scale are allowed.
    """
    _activation_function = staticmethod(act.unscaled_gaussian_bump)
    allowed_keys = frozenset({'inverse_scale'})
    allows_multiple_weights_and_biases = True

    @classmethod
    def from_config(cls, in_size:int, out_size:int, num_splits:int=1, *, key:jax.Array, is_first_layer:bool, **activation_kwargs):
        """from_config create a layer from hyperparameters

        :param in_size: size of the input
        :param out_size: size of the output
        :param num_splits: ignored, defaults to 1
        :param key: key for random number generator (keyword only)
        :param is_first_layer: ignored (keyword only)
        :param inverse_scale: value of the inverse scale parameter in the Gaussian activation e^(-|inverse_scale*x|^2)

        :raises: ValueError if any other activation_kwargs than 'inverse_scale' are provided

        :return: an INR layer with Gaussian activation function and with weights initialized according to the Glorot/Xavier uniform initialization scheme
        """
        # according to the Beyond Periodicity paper by Ramasinghe and Lucey, 
        # this one should be rather robust to what initialization scheme is used.
        # here we'll use Glorot/Xavier uniform
        activation_kwargs = cls._check_keys(activation_kwargs)
        key_gen = key_generator(key)
        lim = jnp.sqrt(6/(in_size+out_size))
        weights = jax.random.uniform(
            key=next(key_gen),
            shape=(out_size, in_size),
            minval=-lim,
            maxval=lim
            )
        
        biases = jax.random.uniform(
            key=next(key_gen),
            shape=(out_size,),
            minval=-1,
            maxval=1
            )
        return cls(weights, biases, **activation_kwargs)


class FinerLayer(INRLayer):
    """
    FINER Layer using variable-periodic activation functions.
    This layer applies a linear transformation followed by a FINER sine activation.
    """

    allowed_keys = frozenset({'w0'})
    allows_multiple_weights_and_biases = False
    _activation_function = staticmethod(act.finer_activation)
    
    @classmethod
    def from_config(cls, in_size: int, out_size: int, num_splits: int = 1, *, key: jax.Array, is_first_layer: bool, **activation_kwargs):
        """
        Initialize FINER layer from hyperparameters
        :param in_size: size of the input
        :param out_size: size of the output
        :param num_splits: ignored, defaults to 1
        :param key: PRNG key for random number generator
        :param is_first_layer: boolean indicating if this is the first layer
        :param w0: scaling factor similar to SIREN (keyword only)
        """
        activation_kwargs = cls._check_keys(activation_kwargs)
        w0 = activation_kwargs['w0']
        w_key, b_key = jax.random.split(key)
        # Weight initialization similar to SIREN
        if is_first_layer:
            lim = 1.0 / in_size
        else:
            lim = jnp.sqrt(6.0 / in_size) / w0
        weights = jax.random.uniform(w_key, shape=(out_size, in_size), minval=-lim, maxval=lim)
        
        # Bias initialization over a larger range to flexibly tune the frequency set
        k = 20  # As per the FINER paper
        biases = jax.random.uniform(b_key, shape=(out_size,), minval=-k, maxval=k)
        return cls(weights, biases, **activation_kwargs)


class PositionalEncodingLayer(eqx.nn.StatefulLayer):
    """ 
    Base class for various kinds of positional encodings. 
    """
    _embedding_matrix: Union[jax.Array, Sequence[jax.Array]]
    _is_learnable: eqx.AbstractClassVar[bool]

    def __init__(self, embedding_matrix):
        self._embedding_matrix = embedding_matrix

    @property
    def embedding_matrix(self):
        """ 
        Get the embedding matrix of the PositionalEncodingLayer
        If not self._is_learnable, apply jax.lax.stop_gradient to prevent the matrix from being changed during training.
        """
        if self._is_learnable:
            return self._embedding_matrix
        else:
            return jax.lax.stop_gradient(self._embedding_matrix)
        
    @abc.abstractmethod
    def out_size(self, in_size:int)->int:
        """ 
        Return the number of output channels given the number of input channels
        :parameter in_size: dimensionality of the input
        :returns: dimensionality of the embedding
        """
        pass

    def is_stateful(self)->bool:
        """ 
        Indicate whether the positional embedding is stateful or not.
        """
        return False


class ClassicalPositionalEncoding(PositionalEncodingLayer):
    """ 
    The standard positional encoding used in NeRF (among other places).
    See https://arxiv.org/pdf/2003.08934v2 (NeRF paper) equation 4 (page 7)
    """
    _is_learnable = False

    @classmethod
    def from_config(cls, num_frequencies:int, frequency_scaling:float=2.):
        """ 
        :parameter num_frequencies: L in equation 4 of the NeRF paper.
            The output of this layer will be 2*num_frequencies*<number of input channels> dimensional
        """
        powers = jnp.arange(num_frequencies, dtype=jnp.int32)
        embedding_matrix = jnp.pow(frequency_scaling, powers)*jnp.pi  # not really the embedding matrix, but we do just apply this to each coordinate as scalar (coordinate) vector (embedding_matrix) multiplication
        return cls(embedding_matrix)
    
    def __call__(self, x, *, key:Optional[jax.Array]=None)->jax.Array:
        frequencies = jax.vmap(lambda coordinate: coordinate*self.embedding_matrix)(x).flatten()
        return jax.vmap(lambda p: jnp.stack((jnp.sin(p), jnp.cos(p)), axis=0), out_axes=0, in_axes=0)(frequencies).flatten()
    
    def out_size(self, in_size):
        """ 
        Return the number of output channels given the number of input channels
        :parameter in_size: dimensionality of the input
        :returns: dimensionality of the embedding
        """
        return 2*self.embedding_matrix.shape[0]*in_size
