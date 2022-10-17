import itertools
from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Tuple

from torch import Tensor, nn

from vital.models.layers import (
    conv2d_activation,
    conv2d_activation_bn,
    conv_transpose2d_activation,
    conv_transpose2d_activation_bn,
    get_nn_module,
)


class Decoder2d(nn.Module):
    """Module making up the decoder half of a convolutional 2D autoencoder."""

    def __init__(
        self,
        output_shape: Tuple[int, int, int],
        blocks: int,
        init_channels: int,
        latent_dim: int,
        activation: str = "ELU",
        use_batchnorm: bool = True,
    ):
        """Initializes class instance.

        Args:
            output_shape: (C, H, W), Shape of the data to reconstruct.
            blocks: Number of upsampling transposed convolution blocks to use.
            init_channels: Number of output feature maps from the last layer before the classifier, used to compute the
                number of feature maps in preceding layers.
            latent_dim: Number of dimensions in the latent space.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
        """
        super().__init__()
        if use_batchnorm:
            conv_block = conv2d_activation_bn
            conv_transpose_block = conv_transpose2d_activation_bn
            batchnorm_desc = "_bn"
        else:
            conv_block = conv2d_activation
            conv_transpose_block = conv_transpose2d_activation
            batchnorm_desc = ""

        # Projection from encoding to bottleneck
        self.feature_shape = (init_channels, output_shape[1] // 2 ** (blocks + 1), output_shape[2] // 2 ** (blocks + 1))
        self.encoding2features = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_fc", nn.Linear(latent_dim, reduce(mul, self.feature_shape))),
                    (f"bottleneck_{activation.lower()}", getattr(nn, activation)()),
                ]
            )
        )

        # Upsampling transposed convolution blocks
        self.features2output = nn.Sequential()
        block_in_channels = init_channels
        for idx, block_idx in enumerate(range(blocks - 1, -1, -1)):
            block_out_channels = init_channels * 2**block_idx
            self.features2output.add_module(
                f"conv_transpose_{activation.lower()}{batchnorm_desc}_{idx}",
                conv_transpose_block(
                    in_channels=block_in_channels, out_channels=block_out_channels, activation=activation
                ),
            )
            self.features2output.add_module(
                f"conv_{activation.lower()}{batchnorm_desc}_{idx}",
                conv_block(in_channels=block_out_channels, out_channels=block_out_channels, activation=activation),
            )
            block_in_channels = block_out_channels

        self.features2output.add_module(
            f"conv_transpose_{activation.lower()}{batchnorm_desc}_{blocks}",
            conv_transpose_block(in_channels=block_in_channels, out_channels=block_in_channels, activation=activation),
        )

        # Classifier
        self.classifier = nn.Conv2d(block_in_channels, output_shape[0], kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            z: (N, ``latent_dim``), Encoding of the input in the latent space.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
        """
        features = self.encoding2features(z)
        features = self.features2output(features.view((-1, *self.feature_shape)))
        out = self.classifier(features)
        return out


class Decoder1d(nn.Module):
    """Module making up the decoder half of a convolutional 1D autoencoder."""

    def __init__(
        self,
        output_shape: Tuple[int, int],
        blocks: int,
        init_channels: int,
        latent_dim: int,
        activation: str = "ELU",
    ):
        """Initializes class instance.

        Args:
            output_shape: (C, L), Shape of the data to reconstruct.
            blocks: Number of upsampling transposed convolution blocks to use.
            init_channels: Number of output channels from the last layer before the regressor, used to compute the
                number of channels in preceding layers.
            latent_dim: Number of dimensions in the latent space.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
        """
        super().__init__()
        channels = [init_channels * 2**block_idx for block_idx in range(blocks - 1, -1, -1)]

        # Projection from encoding to bottleneck
        self.feature_shape = (channels[0], output_shape[1] // 2**blocks)
        self.encoding2features = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_fc", nn.Linear(latent_dim, reduce(mul, self.feature_shape))),
                    (f"bottleneck_{activation.lower()}", getattr(nn, activation)()),
                ]
            )
        )

        def _upsampling_block(block_in_channels: int, block_out_channels: int) -> nn.Module:
            return nn.Sequential(
                OrderedDict(
                    [
                        ("pad", nn.ReflectionPad1d(1)),
                        ("conv", nn.Conv1d(block_in_channels, block_out_channels, kernel_size=3, stride=1)),
                        (activation.lower(), get_nn_module(activation)),
                        ("upsample", nn.Upsample(scale_factor=2, mode="linear")),
                    ]
                )
            )

        # Upsampling convolution blocks
        self.features2output = nn.Sequential()
        for block_idx, (block_in_channels, block_out_channels) in enumerate(itertools.pairwise(channels), start=1):
            self.features2output.add_module(
                f"upsampling_block_{block_idx}", _upsampling_block(block_in_channels, block_out_channels)
            )
        # Last upsampling block does not halve the number of channels, to leave enough for the regressor
        self.features2output.add_module(f"upsampling_block_{blocks}", _upsampling_block(channels[-1], channels[-1]))

        # Regressor
        self.regressor = nn.Sequential(
            nn.ReflectionPad1d(1), nn.Conv1d(channels[-1], output_shape[0], kernel_size=3, stride=1)
        )

    def forward(self, z: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            z: (N, ``latent_dim``), Encoding of the input in the latent space.

        Returns:
            (N, ``out_channels``, ``out_length``), Reconstructed input.
        """
        features = self.encoding2features(z)
        features = features.view((-1, *self.feature_shape))
        features = self.features2output(features)
        out = self.regressor(features)
        return out
