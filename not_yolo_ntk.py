from jax.nn import initializers
import numpy as np
import jax
import flax
import optax
import neural_tangents as nt
from typing import Callable, Sequence, Any, Optional, Union
import flax.linen as nn
import jax.numpy as jnp
from jax import random, jit, vmap
from inr_utils.images import make_lin_grid
from common_jax_utils import key_generator

key_gen = key_generator(random.PRNGKey(0))

# MLP
class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu
    final_layer_sigmoid: bool = False

    @nn.compact
    def __call__(self, x):
        # x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        if self.final_layer_sigmoid:
            x = nn.sigmoid(x)

        return x


# MLP with initial mapping with fourier features
class FFN(nn.Module):
    features: Sequence[int]
    B: jnp.array
    activation: Callable = nn.relu
    final_layer_sigmoid: bool = False

    @nn.compact
    def __call__(self, x):
        x = input_mapping_fourier(x, self.B)
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        if self.final_layer_sigmoid:
            x = nn.sigmoid(x)
        return x


def input_mapping_fourier(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.0 * jnp.pi * x) @ B.T
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class SIREN(nn.Module):
    features: Sequence[int]
    first_omega_0: float = 30
    hidden_omega_0: float = 30
    activation: Callable = jnp.sin
    input_dim: int = 1
    outermost_linear: bool = True

    @nn.compact
    def __call__(self, x):

        # first layer initialization is different than others
        feat_in = self.input_dim
        x = self.activation(
            self.first_omega_0
            * nn.Dense(
                self.features[0],
                kernel_init=my_uniform(scale=1 / feat_in),
                bias_init=my_uniform(scale=1 / feat_in),
            )(x)
        )

        # rest of the layers
        feat_in = self.features[0]
        for feat in self.features[1:-1]:
            x = self.activation(
                self.hidden_omega_0
                * nn.Dense(
                    feat,
                    kernel_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                    bias_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                )(x)
            )
            feat_in = feat

        # final layer
        if self.outermost_linear:
            x = nn.Dense(
                self.features[-1], kernel_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0), bias_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
            )(x)
        else:
            x = self.activation(
                self.hidden_omega_0
                * nn.Dense(
                    self.features[-1],
                    kernel_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                    bias_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                )(x)
            )

        return x


def my_uniform(scale=1e-2, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return random.uniform(key, shape, dtype, -1, 1) * scale

    return init


def complex_gabor_wavelet(*x: jax.Array, s0:Union[float, jax.Array], w0: Union[float, jax.Array])->jax.Array:
    """ 
    Implements the n-dimensional WIRE activation function as per the 2D extension.
    :parameter x: one jax.Array per dimension
        should all have the same size
    :parameter s0: inverse scale parameter of the wavelet (s_0 in the paper)
    :parameter w0: frequency scale parameter of the wavelet (\omega_0 in the paper)
    :returns: the computed wavelet as a JAX array
    """
    # Frequency modulation ( w0 applies only to the first coordinate)
    freq = jnp.exp(1j * w0 * x[0])
    # Radial part: Gaussian envelope
    x = jnp.stack(x, axis=0)
    squared_abs = jnp.mean(jnp.square(jnp.abs(x)), axis=0)  # mean instead of sum so that we don't have to change the initialization scheme
    gaussian_envelope = jnp.exp(-jnp.square(s0)*squared_abs)
    return freq * gaussian_envelope


class RealWIREInitializer():  # Subclass from _Initializer
    def __init__(self) -> None:
        pass


    def __call__(self, key, shape: Sequence[int], dtype=jnp.float32) -> jnp.ndarray:
        in_size = np.prod(shape[:-1])  # For simplicity we assume last dimension is out size
        lim = jnp.sqrt(1./in_size)
        real_part = jax.random.uniform(next(key_gen),shape, minval=-lim, maxval=lim)
        return real_part 

class BiasRealWIREInitializer():
    def __init__(self) -> None:
        pass
    def __call__(self, key, shape: Sequence[int], dtype=jnp.float32) -> jnp.ndarray:
        in_size = np.prod(shape[:-1])  # For simplicity we assume last dimension is out size
        real_part = jax.random.uniform(next(key_gen),shape, minval=-1, maxval=1)
        return real_part


class ComplexWIREInitializer():  # Subclass from _Initializer
    def __init__(self, w0: float):
        self.w0 = w0

    def __call__(self, key, shape: Sequence[int], dtype=jnp.complex64) -> jnp.ndarray:
        in_size = np.prod(shape[:-1])  # For simplicity we assume last dimension is out size
        lim = jnp.sqrt(6./in_size)/self.w0
        real_part = jax.random.uniform(next(key_gen),shape, minval=-lim, maxval=lim)
        imag_part = jax.random.uniform(next(key_gen), shape, minval=-lim, maxval=lim)
        return real_part + 1j * imag_part


class BiasComplexWIREInitializer():  # Subclass from _Initializer
    def __init__(self) -> None:
        pass

    def __call__(self, key, shape: Sequence[int], dtype=jnp.complex64) -> jnp.ndarray:
        in_size = np.prod(shape[:-1])  # For simplicity we assume last dimension is out size
        lim = jnp.sqrt(2)
        real_part = jax.random.uniform(next(key_gen),shape, minval=-lim, maxval=lim)
        imag_part = jax.random.uniform(next(key_gen), shape, minval=-lim, maxval=lim)
        return real_part + 1j * imag_part


class WIRE(nn.Module):
    features: Sequence[int]
    w0: float = 30
    s0: float = 15
    activation: Callable = complex_gabor_wavelet
    input_dim: int = 1
    outermost_linear: bool = True

    @nn.compact
    def __call__(self, x):

        # first layer initialization is different than others
        feat_in = self.input_dim
        x = self.activation(
            nn.Dense(
                self.features[0],
                kernel_init=RealWIREInitializer(),
                bias_init=BiasRealWIREInitializer(),
            )(x),
            s0=self.s0,
            w0=self.w0

        )

        # rest of the layers
        feat_in = self.features[0]
        for feat in self.features[1:-1]:
            x = self.activation(
                 nn.Dense(
                    feat,
                    kernel_init=ComplexWIREInitializer(self.w0),
                    bias_init=BiasComplexWIREInitializer()
                )(x),
                s0=self.s0,
                w0=self.w0

            )
            feat_in = feat

        # final layer
        if self.outermost_linear:
            x = nn.Dense(
                    self.features[-1],
                    kernel_init=ComplexWIREInitializer(self.w0),
                    bias_init=BiasComplexWIREInitializer()
            )(x)


        else:
            x = self.activation(
                nn.Dense(
                    self.features[-1],
                    kernel_init=ComplexWIREInitializer(self.w0),
                    bias_init=BiasComplexWIREInitializer()
                )(x),
                s0=self.s0,
                w0=self.w0)

        return jnp.real(x)




def get_ntk_fns(model, xs):
    

    def apply_fn(params, x):
        return model.apply(params, x, mutable=["batch_stats"])[0]

    kwargs = dict(
        f=apply_fn,
        trace_axes=(),
        vmap_axes=0
    )
      # Different NTK implementations
    jacobian_contraction = jit(nt.empirical_ntk_fn(**kwargs, implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION))
    ntvp = jit(nt.empirical_ntk_fn(**kwargs, implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS))
    str_derivatives = jit(nt.empirical_ntk_fn(**kwargs, implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES))
    auto = jit(nt.empirical_ntk_fn(**kwargs, implementation=nt.NtkImplementation.AUTO))

  # Parameters \theta
    params = model.init(next(key_gen), xs)
    return params, (jacobian_contraction, ntvp, str_derivatives, auto)

def prep_x_ys(n = 10):

    locations = make_lin_grid(0., 1., (n, n))
    location_dims = locations.shape
    in_channels = location_dims[-1]
    out_channels = 2
    flattened_locations = locations.reshape(-1, in_channels)
    #rgb_vals = random.uniform(next(key_gen), (flattened_locations.shape[0], out_channels), minval=0, maxval=1)
    rgb_vals = jnp.sin(2*jnp.pi* flattened_locations)


    return flattened_locations, in_channels, rgb_vals, out_channels


def comparing_ntk_functions():
    flattened_locations, in_channels, rgb_vals, out_channels = prep_x_ys()
   
    siren = SIREN(input_dim=in_channels, features=[128, 128, out_channels]) 
    
    params, (ntk_fn_jacobian_contraction, ntk_fn_ntvp, ntk_fn_str_derivatives, ntk_fn_auto) = get_ntk_fns(model = siren, xs=flattened_locations)

    k_1 = ntk_fn_jacobian_contraction(flattened_locations, flattened_locations, params)
    k_2 = ntk_fn_ntvp(flattened_locations, flattened_locations, params)
    k_3 = ntk_fn_str_derivatives(flattened_locations, flattened_locations, params)
    k_4 = ntk_fn_auto(flattened_locations, flattened_locations, params)

    print(f"{k_1.shape=}")
    print(f"{k_2.shape=}")
    print(f"{k_3.shape=}")
    print(f"{k_4.shape=}")


def linearize():
    flattened_locations, in_channels, rgb_vals, out_channels = prep_x_ys()
    model = SIREN(input_dim=in_channels, features=[128, 128, out_channels]) 
    flattened_locations, in_channels, rgb_vals, out_channels = prep_x_ys()
    params = model.init(next(key_gen), flattened_locations)
    
    def apply_fn(params, x):
        return model.apply(params, x, mutable=["batch_stats"])[0]

    lin_model = nt.linearize(apply_fn, params) 
    return lin_model, params


@jit
def mse(y, y_pred):
    return jnp.mean(jnp.square(y_pred - y))



    
def use_ntk_to_predict():
    flattened_locations, in_channels, rgb_vals, out_channels = prep_x_ys()
    test_flattened_locations, _ , test_rgb_vals, _ = prep_x_ys(n=100)
    model = SIREN(input_dim=in_channels, features=[512, 512, out_channels]) 
    params = model.init(next(key_gen), flattened_locations)
    
    def apply_fn(params, x):
        return model.apply(params, x, mutable=["batch_stats"])[0]


    kernel_fn = nt.empirical_kernel_fn(apply_fn)

    ntk_train_train = kernel_fn(flattened_locations, None, "ntk", params)
    ntk_test_train = kernel_fn(test_flattened_locations, flattened_locations, "ntk", params)
    
    mse_predictor = nt.predict.gradient_descent_mse(ntk_train_train, rgb_vals)
    
    t = 5.
    y_train_0 = apply_fn(params, flattened_locations)
    y_test_0 = apply_fn(params, test_flattened_locations)
    y_train_t, y_test_t = mse_predictor(t, y_train_0, y_test_0, ntk_test_train)
    
    print(f"{mse(rgb_vals, y_train_0)=}")
    print(f"{mse(rgb_vals, y_train_t)=}")
    
    print(f"{mse(test_rgb_vals, y_test_0)=}")
    print(f"{mse(test_rgb_vals, y_test_t)=}")






if __name__ == "__main__":
    #initializer = nn.initializers.xavier_uniform()(next(key_gen), (2,3), jnp.complex64)
    #print(f"{initializer=}")
    #print(f"{initializer.shape=}")
    #print(f"{initializer.dtype=}")
    flattened_locations, in_channels, rgb_vals, out_channels = prep_x_ys()
    
    model = WIRE(
        w0=20,
        s0=15,
        features=[2,3,3,1]
    )

    params = model.init(random.PRNGKey(0), flattened_locations)
    params, (ntk_fn_jacobian_contraction, ntk_fn_ntvp, ntk_fn_str_derivatives, ntk_fn_auto) = get_ntk_fns(model = model, xs=flattened_locations)

    k_1 = ntk_fn_jacobian_contraction(flattened_locations, flattened_locations, params)
    k_2 = ntk_fn_ntvp(flattened_locations, flattened_locations, params)
    k_3 = ntk_fn_str_derivatives(flattened_locations, flattened_locations, params)
    k_4 = ntk_fn_auto(flattened_locations, flattened_locations, params)

    print(f"{k_1.shape=}")
    print(f"{k_2.shape=}")
    print(f"{k_3.shape=}")
    print(f"{k_4.shape=}")
    
    


