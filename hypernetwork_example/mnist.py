import jax
from jax import numpy as jnp
from torchvision import datasets
from torch.utils.data import DataLoader
import PIL


def ToJaxArray(pic:PIL.Image):
    """ 
    Takes an MNIST image (28x28 PIL.Image) and
    * makes it into a jax.Array of dtype jnp.float32
    * adds a channels dimension so it becomes (1, 28, 28)
    * scales to [0., 1.]
    """
    return jnp.asarray(pic, dtype=jnp.float32)[None, ...]/255.

training_data = datasets.MNIST(
    root="example_data",
    train=True,
    download=True,
    transform=ToJaxArray
)

validation_data = datasets.MNIST(
    root="example_data",
    train=False,
    download=True,
    transform=ToJaxArray
)

def collate(batch:list[tuple[jax.Array, int]])->jax.Array:
    """ 
    Takes a list of tuples of (datapoint, label) and stacks the datapoints along axis 0
    """
    return jnp.stack([datapoint for datapoint, label in batch], axis=0)


def get_train_loader(batch_size:int, shuffle:bool):
    """ 
    Get the train dataloader for mnist
    :param batch_size: size of the batches that are to be returned by the dataloader
    :param shuffle: whether the dataloader should shuffle the images before every epoch
    
    NB if shuffle is set to True, the results of a training loop will no longer be deterministic
    given a jax prng key as the randomness in the DataLoader does not come from jax.random
    """
    return DataLoader(
        dataset=training_data,
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=True
    )

def get_validation_loader(batch_size:int, shuffle:bool):
    """ 
    Get the train dataloader for mnist
    :param batch_size: size of the batches that are to be returned by the dataloader
    :param shuffle: whether the dataloader should shuffle the images before every epoch
    """
    return DataLoader(
        dataset=validation_data,
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=True
    )
