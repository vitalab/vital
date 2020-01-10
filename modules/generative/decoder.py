from collections import OrderedDict

import torch
from torch import Tensor
from torch import nn

from vital.modules.layers import conv3x3, conv_transpose3x3


class Decoder(nn.Module):
    """Module making up the decoder half of a convolutional autoencoder."""

    def __init__(self, image_size: (int, int),
                 channels: int,
                 blocks: int,
                 init_channels: int,
                 code_length: int):
        """
        Args:
            image_size: size of the output segmentation groundtruth for each axis.
            channels: number of channels of the image to reconstruct.
            blocks: number of upsampling transposed convolution blocks to use.
            init_channels: number of output feature maps from the last layer before the classifier, used to compute the
                           number of feature maps in preceding layers.
            code_length: number of dimensions in the latent space.
        """
        super().__init__()

        # Projection from encoding to bottleneck
        self.feature_maps = init_channels
        self.bottleneck_size = (image_size[0] // 2 ** (blocks + 1),
                                image_size[1] // 2 ** (blocks + 1))
        self.bottleneck_fc = nn.Sequential(OrderedDict([
            ('bottleneck_fc', nn.Linear(code_length,
                                        self.bottleneck_size[0] * self.bottleneck_size[1] * init_channels)),
            ('bottleneck_elu', nn.ELU(inplace=True))
        ]))

        # Upsampling transposed convolution blocks
        self.features = nn.Sequential()
        block_in_channels = init_channels
        for idx, block_idx in enumerate(reversed(range(blocks))):
            block_out_channels = init_channels * 2 ** block_idx

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

    def forward(self, z: Tensor) -> Tensor:
        """ Defines the computation performed at every call.

        Args:
            z: (N, ``code_length``), encoding of the input in the latent space.

        Returns:
            y_hat: (N, ``channels``, H, W), raw, unnormalized scores for each class in the input's reconstruction.
        """
        features = self.bottleneck_fc(z)
        features = torch.reshape(features, (-1, self.feature_maps, *self.bottleneck_size))
        features = self.features(features)
        out = self.classifier(features)
        return out
