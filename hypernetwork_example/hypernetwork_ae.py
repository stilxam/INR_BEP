import sys
sys.path.append('../')
from typing import Optional

import jax
import equinox as eqx
import model_components as mc

from common_jax_utils import key_generator
from model_components.hypernetwork_components import FullINRLayerAssemblyLayer, HiddenINRLayerAssemblyLayer
from model_components.inr_layers import INRLayer, Linear

class Hypernetwork(eqx.Module):
    """ 
    Example of a hypernetwork
    NB this is probably not a great implementation
    """
    conv_net: eqx.Module
    shared_mlp: eqx.Module
    decoder_heads: list[eqx.Module]
    assembly_layers: list[eqx.Module]

    def __init__(
            self,
            in_features:int,
            conv_features:int,
            shared_features:int,
            mlp_hidden_features:int,
            mlp_depth:int,
            inr_hidden_size:int,
            inr_depth:int,
            low_rank:int,
            kernel_size:int,
            groups:int,
            layer_type:type[INRLayer],
            layer_kwargs:dict,
            *,
            key:jax.Array
            ):
        """ 
        :param in_features: number of channels of a data point
        :param conv_features: base number of channels in the convolutional part
            deeper into the convolutional network this gets multiplied by 2 and then again by 2
        :param shared_features: output size of the shared mlp
        :param mlp_hidden_features: hidden size of the shared mlp
        :param mlp_depth: depth of the shared MLP
        :param inr_hidden_size: hidden size of the resulting inr
        :param inr_depth: depth of the resulting inr
        :param low_rank: rank of the matrices in the low rank factorization
        :param kernel_size: kernel size of the convolutional block
        :param groups: number of groups used in group normalization
        :param layer_type: type of INRLayer used for the resulting INR
        :param layer_kwargs: activation_kwargs for the resulting INRLayers
        :param key: prng key 
        """
        inr_in_size = 2  # 2-d image
        inr_out_size = in_features
        key_gen = key_generator(key)
        self.conv_net = ConvNet(
            in_features=in_features,
            conv_features=conv_features,
            kernel_size=kernel_size,
            groups=groups,
            key=next(key_gen)
        )
        self.shared_mlp = eqx.nn.MLP(
            in_size=4*conv_features,
            out_size=shared_features,
            width_size=mlp_hidden_features,
            depth=mlp_depth,
            key=next(key_gen),
            activation=jax.nn.gelu,
            final_activation=jax.nn.gelu,
        )

        decoder_heads = []
        assembly_layers = []

        # first layer doesn't require factorization because inr_in_size is typically very low (coordinates)
        num_features = inr_in_size*inr_hidden_size + inr_hidden_size  # weight matrix + bias vector
        decoder_heads.append(eqx.nn.Linear(
            in_features=shared_features,
            out_features=num_features,
            key=next(key_gen)
        ))
        assembly_layers.append(FullINRLayerAssemblyLayer(
            in_size=inr_in_size,
            out_size=inr_hidden_size,
            layer_type=layer_type,
            layer_kwargs=layer_kwargs
        ))

        # hidden inr layers will use factorization
        num_features = 2*low_rank*inr_hidden_size + inr_hidden_size  # the two matrices in the low rank factorization and the bias
        for _ in range(inr_depth-1):
            decoder_heads.append(eqx.nn.Linear(
                in_features=shared_features,
                out_features=num_features, 
                key=next(key_gen)
            ))
            assembly_layers.append(HiddenINRLayerAssemblyLayer(
                in_size=inr_hidden_size,
                out_size=inr_hidden_size,
                low_rank=low_rank,
                layer_type=layer_type,
                layer_kwargs=layer_kwargs,
                key=next(key_gen),
                initial_inv_scale=7.9,
                minimal_inv_scale=.1
            ))

        # final layer doesn't require low rank factorization as the output of the inr is low dimensional anyway
        num_features = inr_hidden_size*inr_out_size + inr_out_size
        decoder_heads.append(eqx.nn.Linear(
            in_features=shared_features,
            out_features=num_features,
            key=next(key_gen)
        ))
        assembly_layers.append(FullINRLayerAssemblyLayer(
            in_size=inr_hidden_size,
            out_size=inr_out_size,
            layer_type=Linear,
        ))

        self.decoder_heads = decoder_heads
        self.assembly_layers = assembly_layers

    def __call__(self, x):
        h = self.conv_net(x)
        h = self.shared_mlp(h)
        inr_layers = [
            assembly_layer(decoder_head(h))
            for assembly_layer, decoder_head in zip(self.assembly_layers, self.decoder_heads)
        ]
        return mc.inr_modules.MLPINR(
            inr_layers[0],
            inr_layers[1:-1],
            inr_layers[-1]
        )
    

class ConvBlock(eqx.Module):
    """ 
    Residual 2D convolutional block for the encoder
    Based on https://arxiv.org/pdf/1603.05027
    """
    conv_0: eqx.Module
    conv_1: eqx.Module
    norm_0: eqx.Module
    norm_1: eqx.Module

    def __init__(
            self,
            feature_size:int, 
            kernel_size:int,
            hidden_size:Optional[int]=None,
            groups:int=1,
            *,
            key:jax.Array,
            ):
        """ 
        :param feature_size: number of input and output features
        :param kernel_size: size of the convolutional kernel
        :param hidden_size: number of hidden features
            if None is provided, hidden_size is set to feature_size
        :param groups: number of groups used in group normalization
            default value of 1 results in layer normalization
        :param key: prng key used for initializing the convolutional layers
        """
        key_gen = key_generator(key)
        if hidden_size is None:
            hidden_size = feature_size
        self.conv_0 = eqx.nn.Conv2d(
            in_channels=feature_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding='same', 
            key=next(key_gen)
        )
        self.conv_1 = eqx.nn.Conv2d(
            in_channels=hidden_size,
            out_channels=feature_size,
            kernel_size=kernel_size,
            padding='same',
            key=next(key_gen)
        )
        self.norm_0 = eqx.nn.GroupNorm(
            groups=groups,
            channels=feature_size
        )
        self.norm_1 = eqx.nn.GroupNorm(
            groups=groups,
            channels=hidden_size
        )
    def __call__(self, x:jax.Array)->jax.Array:
        h = self.norm_0(x)
        h = jax.nn.gelu(h)
        h = self.conv_0(h)
        h = self.norm_1(h)
        h = jax.nn.gelu(h)
        h = self.conv_1(h)
        return x+h

class ConvNet(eqx.Module):
    input_layer: eqx.Module
    conv_00: eqx.Module
    conv_01: eqx.Module
    down_0: eqx.Module
    norm_0: eqx.Module
    conv_10: eqx.Module
    conv_11: eqx.Module
    down_1: eqx.Module
    norm_1: eqx.Module
    conv_20: eqx.Module
    conv_21: eqx.Module
    norm_2: eqx.Module

    def __init__(
            self,
            in_features:int,
            conv_features:int,
            kernel_size:int,
            groups:int=1,
            *,
            key:jax.Array
            ):
        """ 
        :param in_features: number of input channels of the images
        :param latent_size: dimensionality of the resulting latent code
        :param conv_features: base number of features used by the convolutional blocks
            every time we down-sample, we double the number of features
        :param kernel_size: size of the convolutional kernels that are used in the res-blocks
        :param groups: number of groups used in group normalization
            default of 1 results in layer normalization
            conv_features should be devisible by groups
        :param key: prng key used for initializing the neural network layers (keyword only)
        """
        key_gen = key_generator(key)
        self.input_layer = eqx.nn.Conv2d(
            in_channels=in_features,
            out_channels=conv_features,
            kernel_size=kernel_size,
            padding="same",
            key=next(key_gen)
        )
        self.conv_00 = ConvBlock(
            conv_features, 
            kernel_size, 
            groups=groups, 
            key=next(key_gen)
        )
        self.conv_01 = ConvBlock(
            conv_features, 
            kernel_size, 
            groups=groups, 
            key=next(key_gen)
        )
        # for the down sampling, we use a convolutional layer with a stride of 2
        self.down_0 = eqx.nn.Conv2d(
            in_channels=conv_features,
            out_channels=2*conv_features,
            kernel_size=2,
            stride=2,
            key=next(key_gen)
        )
        self.norm_0 = eqx.nn.GroupNorm(
            groups=groups,
            channels=2*conv_features,
        )

        # second level of convolutions
        self.conv_10 = ConvBlock(
            2*conv_features, 
            kernel_size, 
            groups=groups, 
            key=next(key_gen)
        )
        self.conv_11 = ConvBlock(
            2*conv_features, 
            kernel_size, 
            groups=groups, 
            key=next(key_gen)
        )

        self.down_1 = eqx.nn.Conv2d(
            in_channels=2*conv_features,
            out_channels=4*conv_features,
            kernel_size=2,
            stride=2,
            key=next(key_gen)
        )
        self.norm_1 = eqx.nn.GroupNorm(
            groups=groups,
            channels=4*conv_features,
        )

        # third level of convolutions
        self.conv_20 = ConvBlock(
            4*conv_features, 
            kernel_size, 
            groups=groups, 
            key=next(key_gen)
        )
        self.conv_21 = ConvBlock(
            4*conv_features, 
            kernel_size, 
            groups=groups, 
            key=next(key_gen)
        )
        self.norm_2 = eqx.nn.GroupNorm(
            groups=groups,
            channels=4*conv_features,
        )

    def __call__(self, x):
        h = jax.nn.gelu(self.input_layer(x))
        h = self.conv_01(self.conv_00(h))
        h = jax.nn.gelu(self.norm_0(self.down_0(h)))
        h = self.conv_11(self.conv_10(h))
        h = jax.nn.gelu(self.norm_1(self.down_1(h)))
        h = self.conv_21(self.conv_20(h))
        # do global average pooling
        return self.norm_2(h.mean(axis=(-1, -2)))  # final two dimensions are spatial dimensions
        
