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
# from distutils.command.install import value
# from operator import ifloordiv
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

    activation_kwargs: dict = eqx.field()  # think w0 for siren or inverse_scale for gaussian
    _activation_function: eqx.AbstractClassVar[Callable]
    allowed_keys: eqx.AbstractClassVar[
        frozenset[Union[str, tuple[str, aux.ANNOT]]]]  # the keys that should be present in activation_kwargs
    allows_multiple_weights_and_biases: eqx.AbstractClassVar[bool]
    learnable_kwarg_keys: tuple

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
                raise ValueError(
                    f"{cls.__name__} does not allow for multiple weights or biases. Got {type(weights)=} and {type(biases)=}.")
            return
        # else
        if w_seq ^ b_seq:  # xor
            raise ValueError(
                f"weights and biases should either both be a tuple/list of jax.Array objects, or both be a jax.Array. Got {type(weights)=} but {type(biases)=}.")
        if w_seq:
            w_len = len(weights)
            b_len = len(biases)
            if w_len != b_len:
                raise ValueError(
                    f"When providing sequences of weights and biases, the sequences should be of equal length (got len(weights)={w_len} but len(biases)={b_len})")

    def activation_function(self, *args):
        """
        Apply the activation function to the input using the kwargs stored in self.activation_kwargs
        """
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return self._activation_function(*args, **kwargs)

    def __init__(self, weights, biases, learnable_kwarg_keys:Optional[tuple[str,...]]=None, **activation_kwargs):
        """
        Initialise an INRLayer from its weights, biases, and activation_kwargs
        """
        self._check_weights_and_biases(weights, biases)
        self.weights = weights
        self.biases = biases
        activation_kwargs = self._check_keys(activation_kwargs)
        self.activation_kwargs = activation_kwargs
        self.learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()

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
        # print(f"{cls.__name__}: {parameters=}")  # this was for debugging
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
    def from_config(cls, in_size: int, out_size: int, num_splits: int = 1, *, key: jax.Array, is_first_layer: bool,
                    **activation_kwargs):
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

    def __call__(self, x: jax.Array, *, key: Optional[jax.Array]=None):
        # key just to have it be compatible with eqx.nn.Sequential
        if isinstance(self.weights, (list, tuple)):
            # when num_splits > 1
            wxb = [w @ x + b for w, b in zip(self.weights, self.biases)]
        else:
            # when num_splits=1
            wxb = (self.weights @ x + self.biases,)
        return self.activation_function(*wxb)

    @classmethod
    def complex_from_config(cls, in_size, out_size, num_splits=1, *, key, is_first_layer, **activation_kwargs):
        """
        Like from_config, but creates a layer with complex weights and biases.
        """
        key_1, key_2 = jax.random.split(key)
        real_part = cls.from_config(in_size=in_size, out_size=out_size, num_splits=num_splits, key=key_1,
                                    is_first_layer=is_first_layer, **activation_kwargs)
        imaginary_part = cls.from_config(in_size=in_size, out_size=out_size, num_splits=num_splits, key=key_2,
                                         is_first_layer=is_first_layer, **activation_kwargs)
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
    learnable_kwarg_keys: tuple


    @classmethod
    def from_config(
            cls,
            in_size: int,
            out_size: int,
            num_splits: int = 1,
            learnable_kwarg_keys: Optional[tuple[str, ...]] = None,
            *,
            key: jax.Array,
            is_first_layer: bool,
            **activation_kwargs
    ):
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
            lim = 1. / in_size  # from https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L630
        else:
            lim = jnp.sqrt(
                6. / in_size) / w0  # from https://arxiv.org/pdf/2006.09661.pdf subsection.3.2 and appendix 1.5 and https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L627

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
        bias_factor = jnp.pi / jnp.sqrt(
            jnp.sum(jnp.square(weight), axis=1))  # from https://arxiv.org/pdf/2102.02611.pdf page 6 third paragaph
        bias = bias_factor * bias


        learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()

        return cls(weight, bias, learnable_kwarg_keys, **activation_kwargs)



    def _activation_function(self, x, w0):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return jnp.sin(w0 * x)

class SinCardLayer(SirenLayer):
    """
    Cardinal Sinusoid from A Sampling Theory Perspective on Activations for Implicit Neural Representations https://arxiv.org/pdf/2402.05427
    #TODO: check what initialization is used in the paper, I initially couldn't find it so I just used siren.
    :param weights: jax.Array containing the weights of the linear part
    :param biases: jax.Array containing the bias of the linear part
    :param w0: w0 hyperparameter as introduced in the SIREN paper by Sitzmann et al.

    No other activation_kwargs than w0 are allowed
    """
    allowed_keys = frozenset({'w0'})
    allows_multiple_weights_and_biases = False
    learnable_kwarg_keys: tuple

    @classmethod
    def from_config(
                  cls,
                  in_size: int,
                  out_size: int,
                  num_splits: int = 1,
                  learnable_kwarg_keys: Optional[tuple[str, ...]] = None,
                  *,
                  key: jax.Array,
                  is_first_layer: bool,
                  **activation_kwargs
          ):
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
                  lim = 1. / in_size  # from https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L630
              else:
                  lim = jnp.sqrt(
                      6. / in_size) / w0  # from https://arxiv.org/pdf/2006.09661.pdf subsection.3.2 and appendix 1.5 and https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L627

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
              bias_factor = jnp.pi / jnp.sqrt(
                  jnp.sum(jnp.square(weight), axis=1))  # from https://arxiv.org/pdf/2102.02611.pdf page 6 third paragaph
              bias = bias_factor * bias
              learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()

              return cls(weight, bias, learnable_kwarg_keys, **activation_kwargs)



    def _activation_function(self, x, w0):
        """
        since the cardinal sinusoid is defined as sin(w0*x)/x, we need to handle the case where x=0
        i wrote a bunch of versions cuz i wasnt sure

        """
        # return jnp.sinc(w0 * x / jnp.pi)
        # return jnp.where(x == 0, 1, jnp.sin(w0 * x) / x)
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return jnp.sinc(w0 * x)




class AdaHoscLayer(INRLayer):
    """

    """
    allowed_keys = frozenset({'w0'})
    allows_multiple_weights_and_biases = False
    learnable_kwarg_keys: tuple

    @classmethod
    def from_config(
               cls,
               in_size: int,
               out_size: int,
               num_splits: int = 1,
               learnable_kwarg_keys: Optional[tuple[str, ...]] = None,
               *,
               key: jax.Array,
               is_first_layer: bool,
               **activation_kwargs
       ):
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
               lim = 1. / in_size  # from https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L630
           else:
               lim = jnp.sqrt(
                   6. / in_size) / w0  # from https://arxiv.org/pdf/2006.09661.pdf subsection.3.2 and appendix 1.5 and https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L627

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
           bias_factor = jnp.pi / jnp.sqrt(
               jnp.sum(jnp.square(weight), axis=1))  # from https://arxiv.org/pdf/2102.02611.pdf page 6 third paragaph
           bias = bias_factor * bias
           learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()

           return cls(weight, bias, learnable_kwarg_keys, **activation_kwargs)


    def _activation_function(self, x, w0):
        """
        since the cardinal sinusoid is defined as sin(w0*x)/x, we need to handle the case where x=0

        """
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.ada_hosc_activation(x, w0)


class HoscLayer(INRLayer):
    """

    """
    allowed_keys = frozenset({'w0'})
    allows_multiple_weights_and_biases = False
    learnable_kwarg_keys: tuple

    @classmethod
    def from_config(
               cls,
               in_size: int,
               out_size: int,
               num_splits: int = 1,
               learnable_kwarg_keys: Optional[tuple[str, ...]] = None,
               *,
               key: jax.Array,
               is_first_layer: bool,
               **activation_kwargs
       ):
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
               lim = 1. / in_size  # from https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L630
           else:
               lim = jnp.sqrt(
                   6. / in_size) / w0  # from https://arxiv.org/pdf/2006.09661.pdf subsection.3.2 and appendix 1.5 and https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L627

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
           bias_factor = jnp.pi / jnp.sqrt(
               jnp.sum(jnp.square(weight), axis=1))  # from https://arxiv.org/pdf/2102.02611.pdf page 6 third paragaph
           bias = bias_factor * bias

           learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()
           return cls(weight, bias, learnable_kwarg_keys, **activation_kwargs)


    def _activation_function(self, x, w0):
        """
        since the cardinal sinusoid is defined as sin(w0*x)/x, we need to handle the case where x=0
        i wrote a bunch of versions cuz i wasnt sure

        """
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.hosc_activation(x, w0)


class RealWIRE(INRLayer):
    """
    RealGaborWavelet INRLayer
    :param weights: jax.Array containing the weights of the frequency and scale components
    :param biases: jax.Array containing the bias of the frequency and scale components
    :param w0: frequency hyperparameter as introduced in the WIRE paper by Vishwanath et al.
    :param s0: spread hyperparameter as introduced in the WIRE paper by Vishwanath et al.
    """
    allowed_keys = frozenset({'w0', 's0'})
    allows_multiple_weights_and_biases = True
    learnable_kwarg_keys: tuple


    @classmethod
    def from_config(cls,
                    in_size,
                    out_size,
                    num_splits=1,
                    learnable_kwarg_keys: Optional[tuple[str, ...]] = None,
                    *,
                    key,
                    is_first_layer,
                    **activation_kwargs
                    ):
        """from_config create a layer from hyperparameters

        :param in_size: size of the input
        :param out_size: size of the output
        :param num_splits: ignored, defaults to 1
        :param key: key for random number generator (keyword only)
        :param is_first_layer: whether this is the first layer in an INR or not (keyword only)

        :raises: ValueError if any other activation_kwargs than 'w0' and 's0' are provided

        :return: a SirenLayer with weights and biases initialized according to the scheme provided in the original SIREN paper
        """
        activation_kwargs = cls._check_keys(activation_kwargs)

        fw_key, subkey = jax.random.split(key)
        fb_key, subkey = jax.random.split(subkey)
        sw_key, subkey = jax.random.split(subkey)
        sb_key, subkey = jax.random.split(subkey)

        # Pytorch initialization
        lim = 1 / jnp.sqrt(in_size)

        frequency_weight = jax.random.uniform(
            key=fw_key,
            shape=(out_size, in_size),
            minval=-lim,
            maxval=lim,
        )

        frequency_bias = jax.random.uniform(
            key=fb_key,
            shape=(out_size,),
            minval=-1,
            maxval=1,
        )

        scale_weight = jax.random.uniform(
            key=sw_key,
            shape=(out_size, in_size),
            minval=-lim,
            maxval=lim,
        )
        scale_bias = jax.random.uniform(
            key=sb_key,
            shape=(out_size,),
            minval=-1,
            maxval=1,
        )

        weights = [frequency_weight, scale_weight]
        biases = [frequency_bias, scale_bias]

        learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()
        return cls(weights, biases, learnable_kwarg_keys, **activation_kwargs)

    def _activation_function(self, *x, w0, s0):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.real_gabor_wavelet(x, s0=s0, w0=w0)


class ComplexWIRE(INRLayer):
    """
    ComplexGaborWavelet INRLayer
    :param weights: jax.Array containing the weights of the linear part
                    Only Float in the first layer, otherwise, Complex64
    :param biases: jax.Array containing the bias of the linear part
    :param w0: frequency hyperparameter as introduced in the WIRE paper by Vishwanath et al.
    :param s0: spread hyperparameter as introduced in the WIRE paper by Vishwanath et al.
    """
    allowed_keys = frozenset({'w0', 's0'})
    allows_multiple_weights_and_biases = True
    # _activation_function = staticmethod(act.complex_gabor_wavelet)
    learnable_kwarg_keys: tuple


    @classmethod
    def from_config(cls,
                    in_size: int,
                    out_size: int,
                    num_splits: int = 1,
                    learnable_kwarg_keys: Optional[tuple[str, ...]] = None,
                    *,
                    key: jax.Array,
                    is_first_layer: bool,
                    **activation_kwargs
                    ):
        """from_config create a layer from hyperparameters

        :param in_size: size of the input
        :param out_size: size of the output
        :param num_splits: n in WIRE n-D
        :param key: key for random number generator (keyword only)
        :param is_first_layer: whether this is the first layer in an INR or not (keyword only)

        :raises: ValueError 'w0' and 's0' are not provided in activation_kwargs

        :return: a SirenLayer with weights and biases initialized according to the scheme provided in the original SIREN paper
        """
        activation_kwargs = cls._check_keys(activation_kwargs)
        w0 = activation_kwargs['w0']

        if is_first_layer:
            lim = 1. / in_size
        else:
            lim = jnp.sqrt(
                6. / in_size
            ) / w0

        key_gen = key_generator(key)

        weights = []
        biases = []

        for _ in range(num_splits):
            weight = jax.random.uniform(
                key=next(key_gen),
                shape=(out_size, in_size),
                minval=-lim,
                maxval=lim,
            )

            bias = jax.random.uniform(
                key=next(key_gen),
                shape=(out_size,),
                minval=-1,
                maxval=1,
            )
            if not is_first_layer:
                # make weight and bias complex
                c_weight = jax.random.uniform(
                    key=next(key_gen),
                    shape=(out_size, in_size),
                    minval=-lim,
                    maxval=lim,
                )

                c_bias = jax.random.uniform(
                    key=next(key_gen),
                    shape=(out_size,),
                    minval=-1,
                    maxval=1,
                )
                weight = jax.lax.complex(weight, c_weight)
                bias = jax.lax.complex(bias, c_bias)

            weights.append(weight)
            biases.append(bias)
        learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()
        return cls(weights, biases, learnable_kwarg_keys, **activation_kwargs)

    def _activation_function(self, *x, w0, s0):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.complex_gabor_wavelet(*x, s0=s0, w0=w0)


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
    def from_config(
            cls,
            in_size: int,
            out_size: int,
            num_splits: int = 1,
            *,
            key: jax.Array,
            is_first_layer: bool,
            **activation_kwargs
    ):
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

        lim = 1. / jnp.sqrt(in_size)
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
    # _activation_function = staticmethod(act.unscaled_gaussian_bump)
    allowed_keys = frozenset({'inverse_scale'})
    allows_multiple_weights_and_biases = True
    learnable_kwarg_keys: tuple


    @classmethod

    def from_config(cls, in_size: int, out_size: int, num_splits: int = 1, learnable_kwarg_keys: Optional[tuple[str, ...]] = None,  *, key: jax.Array, is_first_layer: bool,
                    **activation_kwargs):
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
        lim = jnp.sqrt(6 / (in_size + out_size))
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
        learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()
        return cls(weights, biases, learnable_kwarg_keys, **activation_kwargs)

    def _activation_function(self, x, inverse_scale):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.unscaled_gaussian_bump(x, inverse_scale=inverse_scale)


class QuadraticLayer(GaussianINRLayer):
    # _activation_function = staticmethod(act.quadratic_activation)
    allowed_keys = frozenset({'a'})
    allows_multiple_weights_and_biases = True
    # learnable_kwarg_keys: tuple

    def _activation_function(self, x, a):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.quadratic_activation(x, a)


class MultiQuadraticLayer(GaussianINRLayer):
    # _activation_function = staticmethod(act.multi_quadratic_activation)
    allowed_keys = frozenset({'a'})
    allows_multiple_weights_and_biases = True
    # learnable_kwarg_keys: tuple
    def _activation_function(self, x, a):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.multi_quadratic_activation(x, a)


class LaplacianLayer(GaussianINRLayer):
    # _activation_function = staticmethod(act.laplacian_activation)
    allowed_keys = frozenset({'a'})
    allows_multiple_weights_and_biases = True
    # learnable_kwarg_keys: tuple

    def _activation_function(self, x, a):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.laplacian_activation(x, a)


class SuperGaussianLayer(GaussianINRLayer):
    # _activation_function = staticmethod(act.super_gaussian_activation)
    allowed_keys = frozenset({'a', "b"})
    allows_multiple_weights_and_biases = True
    # learnable_kwarg_keys: tuple

    def _activation_function(self, x, a, b):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.super_gaussian_activation(x, a, b)


class ExpSinLayer(GaussianINRLayer):
    # _activation_function = staticmethod(act.exp_sin_activation)
    allowed_keys = frozenset({'a'})
    allows_multiple_weights_and_biases = True
    # learnable_kwarg_keys: tuple

    def _activation_function(self, x, a):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.exp_sin_activation(x, a)



class FinerLayer(INRLayer):
    """
    FINER Layer using variable-periodic activation functions.
    This layer applies a linear transformation followed by a FINER sine activation.
    """

    allowed_keys = frozenset({'w0'})
    allows_multiple_weights_and_biases = False
    # _activation_function = staticmethod(act.finer_activation)
    learnable_kwarg_keys: tuple


    @classmethod
    def from_config(cls, in_size: int, out_size: int, num_splits: int = 1, learnable_kwarg_keys: Optional[tuple[str, ...]] = None, *, key: jax.Array, is_first_layer: bool, bias_k:float=1.0, **activation_kwargs):

        """
        Initialize FINER layer from hyperparameters
        :param in_size: size of the input
        :param out_size: size of the output
        :param num_splits: ignored, defaults to 1
        :param key: PRNG key for random number generator
        :param is_first_layer: boolean indicating if this is the first layer
        :param bias_k: the k for the initialization scheme
        :param w0: scaling factor similar to SIREN (keyword only)
        """
        activation_kwargs = cls._check_keys(activation_kwargs)
        w0 = activation_kwargs['w0']
        k = bias_k
        w_key, b_key = jax.random.split(key)
        # Weight initialization similar to SIREN
        if is_first_layer:
            lim = 1.0 / in_size
        else:
            lim = jnp.sqrt(6.0 / in_size) / w0
        weights = jax.random.uniform(w_key, shape=(out_size, in_size), minval=-lim, maxval=lim)

        # Bias initialization over a larger range to flexibly tune the frequency set
        biases = jax.random.uniform(b_key, shape=(out_size,), minval=-k, maxval=k)
        learnable_kwarg_keys = learnable_kwarg_keys if learnable_kwarg_keys is not None else tuple()


        return cls(weights, biases, learnable_kwarg_keys, **activation_kwargs)

    def _activation_function(self, x, w0):
        kwargs = {
            key: jax.lax.stop_gradient(value) if key not in self.learnable_kwarg_keys else value
            for key, value in self.activation_kwargs.items()
        }
        return act.finer_activation(x, w0)


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
    def out_size(self, in_size: int) -> int:
        """
        Return the number of output channels given the number of input channels
        :parameter in_size: dimensionality of the input
        :returns: dimensionality of the embedding
        """
        pass

    def is_stateful(self) -> bool:
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
    def from_config(cls, num_frequencies: int, frequency_scaling: float = 2.):
        """
        :parameter num_frequencies: L in equation 4 of the NeRF paper.
            The output of this layer will be 2*num_frequencies*<number of input channels> dimensional
        """
        powers = jnp.arange(num_frequencies, dtype=jnp.int32)
        embedding_matrix = jnp.pow(frequency_scaling,
                                   powers) * jnp.pi  # not really the embedding matrix, but we do just apply this to each coordinate as scalar (coordinate) vector (embedding_matrix) multiplication
        return cls(embedding_matrix)

    def __call__(self, x, *, key: Optional[jax.Array] = None) -> jax.Array:
        frequencies = jax.vmap(lambda coordinate: coordinate * self.embedding_matrix)(x).flatten()
        result = jax.vmap(lambda p: jnp.stack((jnp.sin(p), jnp.cos(p)), axis=0), out_axes=0, in_axes=0)(
            frequencies).flatten()
        return jnp.concatenate([x, result], axis=0)  # concatinate the original input to the frequency outputs

    def out_size(self, in_size):
        """
        Return the number of output channels given the number of input channels
        :parameter in_size: dimensionality of the input
        :returns: dimensionality of the embedding
        """
        return 2 * self.embedding_matrix.shape[0] * in_size + in_size

class TridentPositionalEncoding(PositionalEncodingLayer):
    """
    Positional encoding described in TRIDENT paper
    see https://arxiv.org/pdf/2311.13610
    """
    _is_learnable = False
    s0: float

    @classmethod
    def from_config(cls, num_frequency: int,  frequency_param: float = 2., scale_param: float = 1.):
        # powers = jnp.arange(num_frequency, dtype=jnp.int32)/
        powers = jnp.linspace(0, 1, num_frequency)
        cls.embedding_matrix = 2* jnp.pi * jnp.pow(frequency_param, powers)
        cls.s0 = scale_param
        return cls

    def __call__(self, x, *, key: Optional[jax.Array]=None)-> jax.Array:
        """
        phi_0(x) = exp(-|s0([x, ..., cos(2pi \sigma^{j/m} x), sin(2pi \sigma^{j/m} x)]^T)|^2 )
        """
        frequencies = jax.vmap(lambda coordinate: coordinate * self.embedding_matrix)(x).flatten()
        trig_freq = jax.vmap(lambda p: jnp.stack((jnp.sin(p), jnp.cos(p)), axis=0), out_axes=0, in_axes=0)(frequencies)
        return jnp.exp(- jnp.square(jnp.abs(self.s0 * trig_freq)))

    def out_size(self, in_size):
        """
        Return the number of output channels given the number of input channels
        :parameter in_size: dimensionality of the input
        :returns: dimensionality of the embedding
        """
        return 2 * self.embedding_matrix.shape[0] * in_size


class IntegerLatticeEncoding(PositionalEncodingLayer):
    # TODO Simon: we need to allow for a scheduler in the train step to increase alpha.
    _is_learnable = False
    _min_alpha_state_index: eqx.nn.StateIndex
    _alpha_state_index: eqx.nn.StateIndex
    _max_alpha_state_index: eqx.nn.StateIndex

    def is_stateful(self):
        return True
    
    def __init__(self, embedding_matrix):
        init_alpha = 1.1 
        max_alpha = jnp.linalg.norm(embedding_matrix, axis=1).max()
        
        self._min_alpha_state_index = eqx.nn.StateIndex(jnp.asarray(init_alpha, dtype=jnp.float32))
        self._alpha_state_index = eqx.nn.StateIndex(jnp.asarray(init_alpha, dtype=jnp.float32))
        self._max_alpha_state_index = eqx.nn.StateIndex(jnp.asarray(max_alpha, dtype=jnp.float32))
        super().__init__(embedding_matrix)

    @classmethod
    def from_config(cls, N, dim_input):
        """
        Generates the integer lattice mapping matrix B

        :parameter N: frequency bound
        :dim_input: input dimension
        :return: the integer lattice mapping matrix
        """

        # all possible values in the range [-N, N]
        grid_ranges = [jnp.arange(-N, N + 1) for _ in range(dim_input)]
        
        # create a grid of all combinations of the values in [-N, N]
        grid_mesh = jnp.meshgrid(*grid_ranges, indexing='ij')
        
        # stack combinations into a matrix
        B = jnp.stack([g.flatten() for g in grid_mesh], axis=-1)

        # apply infinity norm constraint ||n||_∞ ≤ N
        mask = jnp.max(jnp.abs(B), axis=1) <= N
        B = B[mask]

        # dentify where all previous elements (n1, ..., n_{j-1} for any j>1) are zero
        zero_prefix_mask = jnp.cumprod(B == 0, axis=1, dtype=jnp.int32)

        # set H, in which a negative component follows a sequence of zeros
        H = jnp.any((zero_prefix_mask[:, :-1] == 1) & (B[:, 1:] < 0), axis=1)

        # remove elements in H
        B = B[~H]

        return cls(B)
    
    def weigh_embedding_matrix(self, state):
        """
        Applies weighing to the rows of the embedding matrix
        Weighing factor alpha is incremented linearly during training 
        until a maximum value is reached  
        """
        embedding_matrix = self.embedding_matrix

        current_alpha = state.get(self._alpha_state_index)
        max_alpha = state.get(self._max_alpha_state_index)
        
        current_alpha = jnp.where(current_alpha < max_alpha, current_alpha, max_alpha)
        
        def weigh_B(B, alpha):
            """
            Assign a weight to the elements of all rows in embedding matrix B,
            where the weighing is evaluated per row

            :parameter B: the original embedding matrix B
            :parameter alpha: the current value of alpha at a given state
            :return: matrix B with weighing applied
            """
            def w_alpha(z):
                """
                Determines a weight for a single row (frequency) of embedding matrix B, 
                depending on the values of alpha and the norm of the row

                :parameter z: a row of embedding matrix B
                :return: the weight to assign to that row
                """
                return jnp.where(
                    alpha - z < 0,
                    0, 
                    jnp.where(alpha-z<=1, 0.5*(1-jnp.cos((alpha-z)*jnp.pi)), 1),
                )
            # collect the norm of each row in B
            norms = jnp.linalg.norm(B, axis=1)

            # weigh the rows
            weights = w_alpha(norms)
            
            # apply the weight of each row to B
            weighted_B = B * weights[:, None]

            return weighted_B
        
        weighted_embedding_matrix = weigh_B(embedding_matrix, current_alpha)

        return weighted_embedding_matrix

    def pruned_model(self, state, target_ratio):
        """
        Replaces the INR's embedding and weights matrices with smaller versions
        that retain the most relevant frequency components of the embedding matrix 
        and the weight vectors that correspond to them
        """

        # TODO add condition for when the pruning happens, dependent on the state
        embedding_matrix = self.embedding_matrix
        weights_matrix = self.weights # not sure how to get the model weights
        
        def prune_B_and_W(B, W, target_ratio):
            """
            :parameter target_ratio: ratio of rows to retain from the embedding matrix,
            e.g. target_ratio = 0.5 means we keep half the number of rows
            :return pruned_B: pruned embedding matrix (retained_rows, in_size)
            :return pruned_W: pruned weights matrix (out_size, 2 * retained_rows)
            """
            # number of frequencies in B
            m = B.shape[0]  

            # sum norms for sine and cosine components in W 
            frequency_importances = jnp.linalg.norm(W[:, :m], axis=0) + jnp.linalg.norm(W[:, m:], axis=0)

            # number of rows to keep
            target_size = jnp.ceil(target_ratio * m).astype(int)

            # get indices of the rows we keep
            kept_indices = jnp.argsort(frequency_importances)[-target_size:]

            pruned_B = B[kept_indices]

            # keep indices in both the sine and cosine halves of W
            pruned_W = jnp.concatenate([W[:, kept_indices], W[:, kept_indices + m]], axis=1)

            return pruned_B, pruned_W
        
        pruned_B, pruned_W = prune_B_and_W(embedding_matrix, weights_matrix, target_ratio)
        
        return pruned_B, pruned_W

    def out_size(self, in_size:int)->int:
        return 2*self.embedding_matrix.shape[0]*in_size

            
    def __call__(self, x:jax.Array, state: eqx.nn.State, *, key:Optional[jax.Array])->tuple[jax.Array, eqx.nn.State]:
        
        embedding_matrix = self.weigh_embedding_matrix(state)

        encoding = jnp.concatenate([jnp.cos(2*jnp.pi*(x @ embedding_matrix.T)), 
                                    jnp.sin(2*jnp.pi*(x @ embedding_matrix.T))], 
                                    axis=-1).flatten()
        
        return encoding, state
    