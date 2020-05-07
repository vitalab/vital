from typing import Union, Tuple

import torch
from torch import Tensor
from torch import nn

from vital.modules.layers import conv3x3_activation


class Encoder(nn.Module):
    """Module making up the encoder half of a convolutional autoencoder."""

    def __init__(self, image_size: Tuple[int, int],
                 in_channels: int,
                 blocks: int,
                 init_channels: int,
                 latent_dim: int,
                 output_distribution: bool = False):
        """
        Args:
            image_size: size of the input segmentation groundtruth for each axis.
            in_channels: number of channels of the image to reconstruct.
            blocks: number of downsampling convolution blocks to use.
            init_channels: number of output feature maps from the first layer, used to compute the number of feature
                           maps in following layers.
            latent_dim: number of dimensions in the latent space.
            output_distribution: whether to add a second head at the end to output ``logvar`` along with the default
                                 ``mu`` head.
        """
        super().__init__()
        self.output_distribution = output_distribution

        # Downsampling convolution blocks
        self.features = nn.Sequential()
        block_in_channels = in_channels
        for block_idx in range(blocks):
            block_out_channels = init_channels * 2 ** block_idx

            self.features.add_module(f'strided_conv_elu{block_idx}',
                                     conv3x3_activation(in_channels=block_in_channels,
                                                        out_channels=block_out_channels,
                                                        stride=2, activation='ELU'))
            self.features.add_module(f'conv_elu{block_idx}',
                                     conv3x3_activation(in_channels=block_out_channels,
                                                        out_channels=block_out_channels,
                                                        activation='ELU'))

            block_in_channels = block_out_channels

        # Bottleneck block
        self.features.add_module('bottleneck_strided_conv_elu',
                                 conv3x3_activation(in_channels=block_in_channels,
                                                    out_channels=init_channels,
                                                    stride=2, activation='ELU'))

        # Fully-connected mapping to encoding
        bottleneck_size = (image_size[0] // 2 ** (blocks + 1),
                           image_size[1] // 2 ** (blocks + 1))
        self.mu_head = nn.Linear(bottleneck_size[0] * bottleneck_size[1] * init_channels,
                                 latent_dim)
        if self.output_distribution:
            self.logvar_head = nn.Linear(bottleneck_size[0] * bottleneck_size[1] * init_channels,
                                         latent_dim)

    def forward(self, y: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Defines the computation performed at every call.

        Args:
            y: (N, ``channels``, H, W), input to reconstruct.

        Returns:
            if not ``output_distribution``:
                z: (N, ``latent_dim``), encoding of the input in the latent space.
            if ``output_distribution``:
                mu: (N, ``latent_dim``), mean of the predicted distribution of the input in the latent space.
                logvar: (N, ``latent_dim``), log variance of the predicted distribution of the input in the latent
                        space.
        """
        features = self.features(y)
        features = torch.flatten(features, 1)
        out = self.mu_head(features)
        if self.output_distribution:
            out = (out, self.logvar_head(features))
        return out
