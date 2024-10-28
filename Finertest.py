# import jax.numpy as jnp

# def generate_alpha(x: jax.Array):
#     """
#     Generate alpha used in the FINER activation.
#     FINER generates alpha as |x| + 1.
#     :param x: input array for alpha generation.
#     :return: alpha array.
#     """
#     return jnp.abs(x) + 1

# def finer_activation(x: jax.Array, omega: float):
#     """
#     FINER activation function: sin(omega * alpha * x)
#     :param x: input array for activation.
#     :param omega: frequency scaling factor (omega).
#     :return: output array after applying variable-periodic activation function.
#     """
#     alpha = generate_alpha(x)
#     return jnp.sin(omega * alpha * x)


# import jax
# import jax.numpy as jnp

# def init_weights(key, fan_in: int, omega: float, is_first: bool):
#     """
#     Initializes weights based on the SIREN/FINER initialization scheme.
#     :param key: random key for JAX.
#     :param fan_in: input dimension size.
#     :param omega: frequency scaling factor (omega).
#     :param is_first: boolean indicating if it's the first layer.
#     :return: initialized weight array.
#     """
#     if is_first:
#         bound = 1.0 / fan_in
#     else:
#         bound = jnp.sqrt(6.0 / fan_in) / omega
#     return jax.random.uniform(key, shape=(fan_in,), minval=-bound, maxval=bound)

# def init_bias(key, out_size: int, k: float = 20):
#     """
#     Initializes bias based on a uniform distribution with a larger range.
#     :param key: random key for JAX.
#     :param out_size: output size (number of neurons in the layer).
#     :param k: scaling factor for bias initialization.
#     :return: initialized bias array.
#     """
#     return jax.random.uniform(key, shape=(out_size,), minval=-k, maxval=k)

# import equinox as eqx

# class FinerLayer(eqx.Module):
#     """
#     FINER Layer using variable-periodic activation functions.
#     This layer applies a linear transformation followed by a FINER sine activation.
#     """
#     weights: jax.Array
#     biases: jax.Array
#     omega: float

#     def __init__(self, in_size: int, out_size: int, key: jax.Array, omega: float = 30.0, is_first: bool = False):
#         """
#         Initializes the FINER layer with the given parameters.
#         :param in_size: input size (number of input features).
#         :param out_size: output size (number of neurons).
#         :param key: JAX random key for initialization.
#         :param omega: frequency scaling factor (omega).
#         :param is_first: boolean indicating if this is the first layer.
#         """
#         key_w, key_b = jax.random.split(key)
#         self.weights = init_weights(key_w, in_size, omega, is_first)
#         self.biases = init_bias(key_b, out_size)
#         self.omega = omega

#     def __call__(self, x):
#         """
#         Forward pass: applies the linear transformation and the FINER activation.
#         :param x: input array.
#         :return: output after applying the FINER activation.
#         """
#         wx_b = jnp.dot(x, self.weights) + self.biases
#         return finer_activation(wx_b, self.omega)



# class Finer(eqx.Module):
#     """
#     Full FINER model with multiple hidden layers.
#     """
#     layers: list

#     def __init__(self, in_size: int, out_size: int, hidden_layers: int = 3, hidden_size: int = 256, key: jax.Array, omega: float = 30.0):
#         """
#         Initialize the FINER network.
#         :param in_size: input size (number of input features).
#         :param out_size: output size (number of output features).
#         :param hidden_layers: number of hidden layers.
#         :param hidden_size: number of neurons in each hidden layer.
#         :param key: JAX random key for initialization.
#         :param omega: frequency scaling factor for the FINER activation.
#         """
#         layers = []
#         keys = jax.random.split(key, num=hidden_layers + 2)

#         # First layer
#         layers.append(FinerLayer(in_size, hidden_size, key=keys[0], omega=omega, is_first=True))

#         # Hidden layers
#         for i in range(hidden_layers):
#             layers.append(FinerLayer(hidden_size, hidden_size, key=keys[i+1], omega=omega))

#         # Output layer (no activation)
#         layers.append(FinerLayer(hidden_size, out_size, key=keys[-1], omega=omega, is_first=False))

#         self.layers = layers

#     def __call__(self, x):
#         """
#         Forward pass through the entire FINER network.
#         :param x: input array.
#         :return: output of the model.
#         """
#         for layer in self.layers:
#             x = layer(x)
#         return x


#         """
#         Different from previous explorations [28, 30, 32] which
# focus on optimizing the weight matrix for manipulating
# frequency candidates with better matching degree, FINER opens a novel way to achieve frequency tuning by modulating the bias vector, or in other words, the phase of
# the variable-periodic activation functions. by increasing the standard deviation during
# bias initialization, thus the spectral bias could be flexibly
# tuned and the expressive power are significantly improved
#         """
