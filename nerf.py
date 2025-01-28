import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from inr_utils.sampling import NeRFSyntheticScenesSampler
from model_components.inr_modules import NeRF
from model_components.inr_layers import SirenLayer
from model_components.inr_layers import ClassicalPositionalEncoding
from inr_utils import nerf_utils
from inr_utils.losses import NeRFLossEvaluator
# Set JAX to run on CPU
# jax.config.update('jax_platform_name', 'cpu')


# Step 1: Initialize the NeRF model
key = jax.random.PRNGKey(0)
nerf_model = NeRF.from_config(
    in_size=(3, 3),
    out_size=(1, 3),
    bottle_size=256,
    block_length=4,
    block_width=512,
    num_blocks=2,
    condition_length=None,
    condition_width=None,
    layer_type=SirenLayer,
    activation_kwargs={"w0": 30.0},
    key=key,
    initialization_scheme=None,
    initialization_scheme_kwargs=None,
    positional_encoding_layer=ClassicalPositionalEncoding.from_config(
        num_frequencies=32
    ),
    # positional_encoding_layer=ClassicalPositionalEncoding.from_config(num_frequencies=3),
    # positional_encoding_layer=None,
    num_splits=1,
    post_processor=None,
)
print(nerf_model)

# Step 2: Initialize the sampler
sampler = NeRFSyntheticScenesSampler(
    split='train',
    name='vase',
    batch_size=10,
    poses_per_batch=10,
    base_path="synthetic_scenes",
    size_limit=-1
)

# Step 3: create a renderer
renderer = nerf_utils.Renderer(
    num_coarse_samples=64,
    num_fine_samples=128,
    near=2.0,
    far=6.0,
    noise_std=1.0,
    white_bkgd=True,
    lindisp=False,
    randomized=True
)

# Step 4: Use the sampler to generate rays and ground truth data
key, sample_key = jax.random.split(key)
ray_origins, ray_directions, sample_key, ground_truth = sampler(sample_key)
prng_keys = jax.random.split(sample_key, num=ray_origins.shape[0])

# # Step 5: render the pixels to see if we get errors
output = jax.vmap(renderer.render_nerf_pixel, (None, 0, 0, 0))(nerf_model, ray_origins, ray_directions, prng_keys)

# # Print the output
print(output)

# # Step 6: create a loss evaluator and see if that works too
loss_evaluator = eqx.filter_jit(NeRFLossEvaluator(
    num_coarse_samples=64,
    num_fine_samples=128,
    near=2.0,
    far=6.0,
    noise_std=1.0,
    white_bkgd=True,
    lindisp=False,
    randomized=True,
    parallel_batch_size=1
))
output = loss_evaluator(
    nerf_model=nerf_model,
    batch=(ray_origins, ray_directions, sample_key, ground_truth)
)
print(output)
