from typing import Callable, Sequence

from matplotlib import pyplot as plt
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core.frozen_dict import freeze
import optax
import neural_tangents as nt
from scipy.sparse.linalg import eigsh
import haiku as hk




def my_uniform(scale=1e-2, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return random.uniform(key, shape, dtype, -1, 1) * scale

    return init


class SIREN(nn.Module):
    """
        An implementation of V. Sitzmann et al. 2020 SIREN in flax
    """
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
                self.features[-1],
                kernel_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
                bias_init=my_uniform(scale=jnp.sqrt(6 / feat_in) / self.hidden_omega_0),
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






class hkSIRENLayer(hk.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.out_f = out_f
        self.b = 1 / in_f if self.is_first else jnp.sqrt(6 / in_f) / w0

    def __call__(self, x):
        x = hk.Linear(
            output_size=self.out_f,
            w_init=hk.initializers.RandomUniform(-self.b, self.b),
        )(x)
        return x + 0.5 if self.is_last else self.w0 * x


class hkSIREN(hk.Module):
    def __init__(self, w0, width, hidden_w0, depth):
        super().__init__()
        self.w0 = w0  # to change the omega_0 of SIREN !!!!
        self.width = width
        self.depth = depth
        self.hidden_w0 = hidden_w0

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = hkSIRENLayer(x.shape[-1], self.width, is_first=True, w0=self.w0)(x)
        x = jnp.sin(x)

        for _ in range(self.depth - 2):
            x = hkSIRENLayer(x.shape[-1], self.width, w0=self.hidden_w0)(x)
            x = jnp.sin(x)

        out = hkSIRENLayer(x.shape[-1], 1, w0=self.hidden_w0, is_last=True)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out






def train_data_loader(key, len_x, batch_size, shuffle=True):
    """
        Creates fake dataset of Sin (period rescaled to fit [-1,1])
    """
    x = jnp.linspace(-1, 1, len_x)
    y = jnp.sin(jnp.pi * x)
    if shuffle:
        shuffle_idx = random.permutation(key, len_x)
        x = x[shuffle_idx]
        y = y[shuffle_idx]

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    while True:
        for i in range(0, len_x, batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]


def test_data_loader(len_x, batch_size, ):
    """
        Same fake dataset, not suffled, only yields a finite amount of  points,
    """
    x = jnp.linspace(-1, 1, len_x)
    y = jnp.sin(jnp.pi * x)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    for i in range(0, len_x, batch_size):
        if i + batch_size < len_x:
            yield x[i:i + batch_size], y[i:i + batch_size]
        else:
            yield x[i:], y[i:]


def fit(model, params, optimizer, opt_state, train_dl, config, epochs, log_every):
    """
        Fits model on dataset
    """

    @jax.jit
    def predict(params, x):
        """
            Makes predictions with model, parameters, and Xs
        """
        return jax.vmap(model.apply, in_axes=(None, 0))(params, x)

    @jax.jit
    def mse(params, x, y):
        """
            MSE/2
        """
        return jnp.mean(jnp.square(predict(params, x) - y)) / 2

    def test(params, config):
        """
            Calculates Test Loss
        """
        test_dl = test_data_loader(config["len_x"], config["batch_size"])
        test_list = []

        for x, y in test_dl:
            l = mse(params, x, y)

            test_list.append(l)

        test_list = jnp.array(test_list)
        mean = jnp.mean(test_list)
        if jnp.isnan(mean):
            raise ValueError(f"Test Loss is NaN")

        return jnp.mean(test_list)

    train_loss_list = []
    test_loss_list = []

    for i in tqdm(range(epochs), desc="Training"):
        x, y = next(train_dl)

        loss, grads = jax.value_and_grad(mse)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        train_loss_list.append(loss)

        if i % log_every == 0:
            test_loss_list.append(test(params, config))

    return params, train_loss_list, test_loss_list



def make_variables(params, model_state):
    return freeze({"params": params, **model_state})


def flatten_kernel(K):
    return K.transpose([0, 2, 1, 3]).reshape([K.shape[0] * K.shape[2], K.shape[1] * K.shape[3]])



def get_ntk_fn(apply_fn, variables, batch_size):
    def apply_params_fn(params, x):
        if "batch_stats" in variables.keys():
            model_state, _ = variables.pop("params")
            apply_vars = make_variables(params, model_state)
            # logits = jax.vmap(apply_fn, in_axes=(None, 0))(apply_vars, x)
        #
            logits = apply_fn(apply_vars, x, train=False)
        else:
            logits = apply_fn(params, x)
            # logits = jax.vmap(apply_fn, in_axes=(None, 0))(params, x)

        return logits

    kernel_fn = nt.batch(
        nt.empirical_kernel_fn(apply_params_fn, vmap_axes=0, implementation=2, trace_axes=()),
        batch_size=batch_size,
        device_count=0,
        store_on_device=False,
    )

    def expanded_kernel_fn(data1, data2, kernel_type, params):
        K = kernel_fn(data1, data2, kernel_type, params)
        return flatten_kernel(K)

    return expanded_kernel_fn

def ntk_eigendecomposition(apply_fn, variables, data, batch_size):
    kernel_fn = get_ntk_fn(apply_fn, variables, batch_size)

    if "params" in variables:
        ntk_matrix = kernel_fn(data, None, "ntk", variables["params"])
    else:
        ntk_matrix = kernel_fn(data, None, "ntk", variables)

    eigvals, eigvecs = eigsh(jax.device_get(ntk_matrix), k=2000)
    eigvals = jnp.flipud(eigvals)
    eigvecs = jnp.flipud(eigvecs.T)

    return eigvals, eigvecs, ntk_matrix




def main(config=None):
    """
        executes training
    """
    if config is None:
        config = {
            "model": "SIREN",
            "first_omega_0": 5,
            "hidden_omega_0": 7,
            "features": [512, 512, 512, 1],
            "optimizer": "adam",
            "learning_rate": 5e-5,
            "len_x": 1000,
            "batch_size": 100,
            "shuffle": True,
            "epochs": 10,
            "seed": 0,
            "log_every": 1
        }

    if config["model"] == "SIREN":
        model = SIREN(
            features=config["features"],
            first_omega_0=config["first_omega_0"],
            hidden_omega_0=config["hidden_omega_0"],

        )
    else:
        model = SIREN(
            features=config["features"],
            first_omega_0=config["first_omega_0"],
            hidden_omega_0=config["hidden_omega_0"],
        )
        ValueError(f"Config model {config['model']} is not a valid model")

    if config["optimizer"] == "adam":
        optimizer = optax.adam(config["learning_rate"])
    else:
        optimizer = optax.adam(config["learning_rate"])
        ValueError(f"Config optimizer {config['optimizer']} is not a valid option")

    key = random.PRNGKey(config["seed"])

    key, subkey = random.split(key)

    train_dl = train_data_loader(subkey, config["len_x"], config["batch_size"], config["shuffle"])

    key, subkey = random.split(key)
    init_param = model.init(subkey, next(train_dl)[0])

    opt_state = optimizer.init(init_param)

    params, training_loss, test_loss = fit(model, init_param, optimizer, opt_state, train_dl, config, config["epochs"],
                                           config["log_every"])

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(training_loss, label="train loss")
    ax[1].plot(test_loss, label="test loss")
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    test_dl = test_data_loader(config["len_x"], config["batch_size"])

    gt_x = jnp.linspace(-1, 1, config["len_x"])
    y = jnp.sin(jnp.pi * gt_x)
    ax.plot(gt_x, y, label="gt", color="red")

    pred_ys = []
    for x, y in test_dl:
        pred_ys.append(model.apply(params, x))

    pred_ys = jnp.concatenate(pred_ys)
    ax.plot(gt_x, pred_ys, label="pred", color="blue")
    plt.legend()
    plt.show()

    # look at how ntk.py does the ntk calculations
    #
    # look at figure_4.py and repeat




def ntk_application():
    config = {
        "model": "SIREN",
        "first_omega_0": 5,
        "hidden_omega_0": 7,
        "features": [512, 512, 512, 1],
        "optimizer": "adam",
        "learning_rate": 5e-5,
        "len_x": 1000,
        "batch_size": 100,
        "shuffle": True,
        "epochs": 10,
        "seed": 0,
        "log_every": 1
    }
    # model = SIREN(
    #     features=config["features"],
    #     first_omega_0=config["first_omega_0"],
    #     hidden_omega_0=config["hidden_omega_0"],
    # )
    # init_param = model.init(random.PRNGKey(0), jnp.ones((1, 2)))


    model_SIREN = hk.without_apply_rng(
        hk.transform(lambda x: hkSIREN(w0=30, width=256, hidden_w0=30, depth=5)(x))
    )
    hk_params = model_SIREN.init(random.PRNGKey(0), jnp.ones((1, 2)))

    DEFAULT_RESOLUTION = 64
    x1 = jnp.linspace(0, 1, DEFAULT_RESOLUTION + 1)[:-1]
    DEFAULT_GRID = jnp.stack(jnp.meshgrid(x1, x1, indexing="ij"), axis=-1)[None, ...]




    print()

    eigvals_model, eigvecs_model, ntk_matrix_model = ntk_eigendecomposition(
        model_SIREN.apply, hk_params, DEFAULT_GRID, 2
    )


if __name__ == "__main__":
    # main()
    ntk_application()