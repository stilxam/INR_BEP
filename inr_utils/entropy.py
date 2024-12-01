

# import sys
# sys.path.append('/home/goshko/INR_BEP')

import jax
import jax.numpy as jnp
from inr_utils.images import load_image_as_array


def dft_rgb_image(image:jax.Array)->jax.Array:
    """
    Compute the DFT of a RGB image stored as an array, 
        taking the DFT for each channel separately, and then stacking the results
    :parameter image: an array representation of an image
    :return: DFT of each color channel, stored as an array of the shape (H, W, (R,G,B))
    """
    fft_r = jnp.fft.fft2(image[:, :, 0])
    fft_g = jnp.fft.fft2(image[:, :, 1])
    fft_b = jnp.fft.fft2(image[:, :, 2])

    fft_image = jnp.stack([fft_r, fft_g, fft_b], axis=-1)
    return fft_image


def compute_normalized_power_spectrum(data):

    # compute power spectrum
    fft_result = jnp.fft.fftn(data)
    power_spectrum = jnp.abs(fft_result) ** 2
    # normalize the power spectrum
    power_spectrum_sum = jnp.sum(power_spectrum)
    normalized_spectrum = power_spectrum / power_spectrum_sum

    return normalized_spectrum


def compute_ref_dist_uniform(normalized_spectrum):
    
    num_elements = normalized_spectrum.size
    # define the uniform distribution
    ref_dist = jnp.full(normalized_spectrum.shape, 1.0 / num_elements)
    return ref_dist

def compute_relative_entropy(normalized_spectrum, ref_dist):

    # compute the relative entropy, adding epsilon to avoid log(0) errors
    epsilon = 1e-10
    relative_entropy = jnp.sum(
        normalized_spectrum * (jnp.log((normalized_spectrum + epsilon)) - jnp.log(ref_dist + epsilon)))
    return relative_entropy



def relative_entropy_uniform(input:jax.Array)->float:
    """
    Compute the relative entropy of a data point (Kullback–Leibler divergence) w.r.t. the uniform distribution 
        Since data tends to be predominantly made up of low frequency components, a high relative entropy
        implies lower complexity
    :parameter input: a jax.Array representation of the input data point
    :return: the relative entropy of the input 
    """

    normalized_spectrum = compute_normalized_power_spectrum(input)

    ref_dist = compute_ref_dist_uniform(normalized_spectrum)

    relative_entropy = compute_relative_entropy(normalized_spectrum, ref_dist)

    return relative_entropy




def relative_entropy_rgb(image:jax.Array)->float:
    """
    Compute the relative entropy of an image by averaging the entropy across the color channels
    :parameter input: a jax.Array representation of an image with shape (H, W, (R, G, B))
    :parameter reference_distribution: a string naming the reference distribution
        ('uniform', ...)
    :return: the relative entropy of the image
    """
    r = relative_entropy_uniform(image[:, :, 0])
    g = relative_entropy_uniform(image[:, :, 1])
    b = relative_entropy_uniform(image[:, :, 2])

    return (r + g + b)/3


parrot = load_image_as_array(
    '/home/goshko/INR_BEP/example_data/parrot.png')


# dft_parrot = dft_rgb_image(parrot)
# e = relative_entropy_ndim(parrot[0], 'uniform')
# entropy_parrot = relative_entropy_rgb(parrot, 'uniform')
# print(entropy_parrot)