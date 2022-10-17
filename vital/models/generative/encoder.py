import itertools
from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Tuple, Union

import torch
from torch import Tensor, nn

from vital.models.layers import conv2d_activation, conv2d_activation_bn, get_nn_module


class Encoder2d(nn.Module):
    """Module making up the encoder half of a convolutional 2D autoencoder."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        blocks: int,
        init_channels: int,
        latent_dim: int,
        activation: str = "ELU",
        use_batchnorm: bool = True,
        output_distribution: bool = False,
    ):
        """Initializes class instance.

        Args:
            input_shape: (C, H, W), Shape of the inputs to encode.
            blocks: Number of downsampling convolution blocks to use.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            latent_dim: Number of dimensions in the latent space.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
            output_distribution: Whether to add a second head at the end to output ``logvar`` along with the default
                ``mu`` head.
        """
        super().__init__()
        self.output_distribution = output_distribution
        if use_batchnorm:
            conv_block = conv2d_activation_bn
            batchnorm_desc = "_bn"
        else:
            conv_block = conv2d_activation
            batchnorm_desc = ""
        strided_conv_kwargs = {"stride": 2}

        # Downsampling convolution blocks
        self.input2features = nn.Sequential()
        block_in_channels = input_shape[0]
        for block_idx in range(blocks):
            block_out_channels = init_channels * 2**block_idx

            self.input2features.add_module(
                f"strided_conv_{activation.lower()}{batchnorm_desc}_{block_idx}",
                conv_block(
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    conv_kwargs=strided_conv_kwargs,
                    activation=activation,
                ),
            )
            self.input2features.add_module(
                f"conv_{activation.lower()}{batchnorm_desc}_{block_idx}",
                conv_block(in_channels=block_out_channels, out_channels=block_out_channels, activation=activation),
            )

            block_in_channels = block_out_channels

        # Bottleneck block
        self.input2features.add_module(
            f"bottleneck_strided_conv_{activation.lower()}{batchnorm_desc}",
            conv_block(
                in_channels=block_in_channels,
                out_channels=init_channels,
                conv_kwargs=strided_conv_kwargs,
                activation=activation,
            ),
        )

        # Fully-connected mapping to encoding
        feature_shape = (init_channels, input_shape[1] // 2 ** (blocks + 1), input_shape[2] // 2 ** (blocks + 1))
        self.mu_head = nn.Linear(reduce(mul, feature_shape), latent_dim)
        if self.output_distribution:
            self.logvar_head = nn.Linear(reduce(mul, feature_shape), latent_dim)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input to reconstruct.

        Returns:
            if not ``output_distribution``:
                - (N, ``latent_dim``), Encoding of the input in the latent space.
            if ``output_distribution``:
                - (N, ``latent_dim``), Mean of the predicted distribution of the input in the latent space.
                - (N, ``latent_dim``), Log variance of the predicted distribution of the input in the latent space.
        """
        features = self.input2features(x)
        features = torch.flatten(features, 1)
        out = self.mu_head(features)
        if self.output_distribution:
            out = (out, self.logvar_head(features))
        return out


class Encoder1d(nn.Module):
    """Module making up the encoder half of a convolutional 1D autoencoder."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        blocks: int,
        init_channels: int,
        latent_dim: int,
        activation: str = "ELU",
        output_distribution: bool = False,
    ):
        """Initializes class instance.

        Args:
            input_shape: (C, L), Shape of the inputs to encode.
            blocks: Number of downsampling convolution blocks to use.
            init_channels: Number of output channels from the first layer, used to compute the number of channels in
                following layers.
            latent_dim: Number of dimensions in the latent space.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
            output_distribution: Whether to add a second head at the end to output ``logvar`` along with the default
                ``mu`` head.
        """
        super().__init__()
        self.output_distribution = output_distribution
        channels = [input_shape[0]] + [init_channels * 2**block_idx for block_idx in range(blocks)]

        def _downsampling_block(block_in_channels: int, block_out_channels: int) -> nn.Module:
            return nn.Sequential(
                OrderedDict(
                    [
                        ("pad", nn.ReflectionPad1d(1)),
                        ("conv", nn.Conv1d(block_in_channels, block_out_channels, kernel_size=3, stride=2)),
                        (activation.lower(), get_nn_module(activation)),
                    ]
                )
            )

        # Downsampling convolution blocks
        self.input2features = nn.Sequential()
        for block_idx, (block_in_channels, block_out_channels) in enumerate(itertools.pairwise(channels), start=1):
            self.input2features.add_module(
                f"downsampling_block_{block_idx}", _downsampling_block(block_in_channels, block_out_channels)
            )

        # Fully-connected heads to output mu (and possibly logvar) from the shared feature encoder
        features_shape = (channels[-1], input_shape[1] // 2**blocks)
        self.mu_head = nn.Linear(reduce(mul, features_shape), latent_dim)
        if self.output_distribution:
            self.logvar_head = nn.Linear(reduce(mul, features_shape), latent_dim)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, ``in_length``), Input to reconstruct.

        Returns:
            if not ``output_distribution``:
                - (N, ``latent_dim``), Encoding of the input in the latent space.
            if ``output_distribution``:
                - (N, ``latent_dim``), Mean of the predicted distribution of the input in the latent space.
                - (N, ``latent_dim``), Log variance of the predicted distribution of the input in the latent space.
        """
        features = self.input2features(x)
        features = torch.flatten(features, 1)
        out = self.mu_head(features)
        if self.output_distribution:
            out = (out, self.logvar_head(features))
        return out
