from collections import OrderedDict

import torch
from torch import nn as nn

from vital.modules.generative.layers import conv3x3, conv_transpose3x3


class Decoder(nn.Module):

    def __init__(self, image_size: (int, int),
                 channels: int,
                 blocks: int,
                 feature_maps: int,
                 code_length: int):
        """ Module making up the decoder half of a convolutional autoencoder.

        Args:
            image_size: size of the output segmentation groundtruth for each axis.
            channels: number of channels of the image to reconstruct.
            blocks: number of upsampling transposed convolution blocks to use.
            feature_maps: factor used to compute the number of feature maps for the convolution layers.
            code_length: number of dimensions in the latent space.
        """
        super().__init__()

        # Projection from encoding to bottleneck
        self.feature_maps = feature_maps
        self.bottleneck_size = (image_size[0] // 2 ** (blocks + 1),
                                image_size[1] // 2 ** (blocks + 1))
        self.bottleneck_fc = nn.Sequential(OrderedDict([
            ('bottleneck_fc', nn.Linear(code_length,
                                        self.bottleneck_size[0] * self.bottleneck_size[1] * feature_maps)),
            ('bottleneck_elu', nn.ELU(inplace=True))
        ]))

        # Upsampling transposed convolution blocks
        self.features = nn.Sequential()
        block_in_channels = feature_maps
        for idx, block_idx in enumerate(reversed(range(blocks))):
            block_out_channels = feature_maps * 2 ** block_idx

            self.features.add_module(f'conv_transpose{idx}', conv_transpose3x3(in_channels=block_in_channels,
                                                                               out_channels=block_out_channels,
                                                                               stride=2, output_padding=1))
            self.features.add_module(f'elu{idx}_0', nn.ELU(inplace=True))
            self.features.add_module(f'conv{idx}', conv3x3(in_channels=block_out_channels,
                                                           out_channels=block_out_channels))
            self.features.add_module(f'elu{idx}_1', nn.ELU(inplace=True))

            block_in_channels = block_out_channels

        # Classifier
        self.classifier = conv_transpose3x3(in_channels=block_in_channels, out_channels=channels,
                                            stride=2, output_padding=1)

    def forward(self, x):
        features = self.bottleneck_fc(x)
        features = torch.reshape(features, (-1, self.feature_maps, *self.bottleneck_size))
        features = self.features(features)
        out = self.classifier(features)
        return out
