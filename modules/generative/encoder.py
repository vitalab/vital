import torch
from torch import nn as nn

from vital.modules.generative.layers import conv3x3


class Encoder(nn.Module):

    def __init__(self, image_size: (int, int),
                 channels: int,
                 blocks: int,
                 feature_maps: int,
                 code_length: int,
                 output_distribution: bool = False):
        """ Module making up the encoder half of a convolutional autoencoder.

        Args:
            image_size: size of the input segmentation groundtruth for each axis.
            channels: number of channels of the image to reconstruct.
            blocks: number of downsampling convolution blocks to use.
            feature_maps: factor used to compute the number of feature maps for the convolution layers.
            code_length: number of dimensions in the latent space.
            output_distribution: whether to add a second head at the end to output ``logvar`` along with the default
                                 ``mu`` head.
        """
        super().__init__()
        self.output_distribution = output_distribution

        # Downsampling convolution blocks
        self.features = nn.Sequential()
        block_in_channels = channels
        for block_idx in range(blocks):
            block_out_channels = feature_maps * 2 ** block_idx

            self.features.add_module(f'strided_conv{block_idx}', conv3x3(in_channels=block_in_channels,
                                                                         out_channels=block_out_channels,
                                                                         stride=2))
            self.features.add_module(f'elu{block_idx}_0', nn.ELU(inplace=True))
            self.features.add_module(f'conv{block_idx}', conv3x3(in_channels=block_out_channels,
                                                                 out_channels=block_out_channels))
            self.features.add_module(f'elu{block_idx}_1', nn.ELU(inplace=True))

            block_in_channels = block_out_channels

        # Bottleneck block
        self.features.add_module('bottleneck_strided_conv', conv3x3(in_channels=block_in_channels,
                                                                    out_channels=feature_maps,
                                                                    stride=2))
        self.features.add_module('bottleneck_elu', nn.ELU(inplace=True))

        # Fully-connected mapping to encoding
        bottleneck_size = (image_size[0] // 2 ** (blocks + 1),
                           image_size[1] // 2 ** (blocks + 1))
        self.mu_head = nn.Linear(bottleneck_size[0] * bottleneck_size[1] * feature_maps,
                                 code_length)
        if self.output_distribution:
            self.logvar_head = nn.Linear(bottleneck_size[0] * bottleneck_size[1] * feature_maps,
                                         code_length)

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        out = self.mu_head(features)
        if self.output_distribution:
            out = (out, self.logvar_head(features))
        return out
