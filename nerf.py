import jax
import jax.numpy as jnp
import optax
from inr_utils.sampling import NeRFSyntheticScenesSampler
from model_components.inr_modules import NeRF
from model_components.inr_layers import SirenLayer
from model_components.inr_layers import ClassicalPositionalEncoding

# Set JAX to run on CPU
# jax.config.update('jax_platform_name', 'cpu')


# Step 1: Initialize the NeRF model
key = jax.random.PRNGKey(0)
nerf_model = NeRF.from_config(
    in_size=(3, 2),
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
    positional_encoding_layer=ClassicalPositionalEncoding,
    # positional_encoding_layer=ClassicalPositionalEncoding.from_config(num_frequencies=3),
    # positional_encoding_layer=None,
    num_splits=1,
    post_processor=None,
    num_coarse_samples=64,
    num_fine_samples=128,
    use_viewdirs=True,
    near=2.0,
    far=6.0,
    noise_std=1.0,
    white_bkgd=True,
    lindisp=False
)

# Step 2: Initialize the sampler
sampler = NeRFSyntheticScenesSampler(
    split='train',
    name='armchair_dataset_small',
    batch_size=10,
    poses_per_batch=10,
    base_path="example_data/nerfdata",
    size_limit=-1
)

# Step 3: Use the sampler to generate rays and ground truth data
key, sample_key = jax.random.split(key)
ray_origins, ray_directions, sample_key, ground_truth = sampler(sample_key)

# Step 4: Perform a forward pass through the NeRF model
randomized = True
output = nerf_model(ray_origins, ray_directions, randomized, key=sample_key)

# Print the output
print(output)
