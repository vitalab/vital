from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from vital.modules.layers import conv3x3_activation, conv_transpose2x2_activation


class Decoder(nn.Module):
    """Module making up the decoder half of a convolutional autoencoder."""

    def __init__(self, image_size: Tuple[int, int],
                 out_channels: int,
                 blocks: int,
                 init_channels: int,
                 latent_dim: int):
        """
        Args:
            image_size: size of the output segmentation groundtruth for each axis.
            out_channels: number of channels of the image to reconstruct.
            blocks: number of upsampling transposed convolution blocks to use.
            init_channels: number of output feature maps from the last layer before the classifier, used to compute the
                           number of feature maps in preceding layers.
            latent_dim: number of dimensions in the latent space.
        """
        super().__init__()

        # Projection from encoding to bottleneck
        self.feature_maps = init_channels
        self.bottleneck_size = (image_size[0] // 2 ** (blocks + 1),
                                image_size[1] // 2 ** (blocks + 1))
        self.bottleneck_fc = nn.Sequential(OrderedDict([
            ('bottleneck_fc', nn.Linear(latent_dim,
                                        self.bottleneck_size[0] * self.bottleneck_size[1] * init_channels)),
            ('bottleneck_elu', nn.ELU(inplace=True))
        ]))

        # Upsampling transposed convolution blocks
        self.features = nn.Sequential()
        block_in_channels = init_channels
        for idx, block_idx in enumerate(reversed(range(blocks))):
            block_out_channels = init_channels * 2 ** block_idx
            self.features.add_module(f'conv_transpose_elu{idx}',
                                     conv_transpose2x2_activation(in_channels=block_in_channels,
                                                                  out_channels=block_out_channels,
                                                                  activation='ELU'))
            self.features.add_module(f'conv_elu{idx}',
                                     conv3x3_activation(in_channels=block_out_channels,
                                                        out_channels=block_out_channels,
                                                        activation='ELU'))
            block_in_channels = block_out_channels

        self.features.add_module(f'conv_transpose_elu{blocks}',
                                 conv_transpose2x2_activation(in_channels=block_in_channels,
                                                              out_channels=block_in_channels,
                                                              activation='ELU'))

        # Classifier
        self.classifier = nn.Conv2d(block_in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            z: (N, ``latent_dim``), encoding of the input in the latent space.

        Returns:
            y_hat: (N, ``channels``, H, W), raw, unnormalized scores for each class in the input's reconstruction.
        """
        features = self.bottleneck_fc(z)
        features = torch.reshape(features, (-1, self.feature_maps, *self.bottleneck_size))
        features = self.features(features)
        out = self.classifier(features)
        return out
