from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Tuple

from torch import Tensor, nn

from vital.modules.layers import conv3x3_bn_activation, conv_transpose2x2_bn_activation


class Decoder(nn.Module):
    """Module making up the decoder half of a convolutional autoencoder."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        out_channels: int,
        blocks: int,
        init_channels: int,
        latent_dim: int,
        use_batchnorm: bool = True,
    ):  # noqa: D205,D212,D415
        """
        Args:
            image_size: Size of the output segmentation groundtruth for each axis.
            out_channels: Number of channels of the image to reconstruct.
            blocks: Number of upsampling transposed convolution blocks to use.
            init_channels: Number of output feature maps from the last layer before the classifier, used to compute the
                number of feature maps in preceding layers.
            latent_dim: Number of dimensions in the latent space.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
        """
        super().__init__()
        batchnorm_desc = "_bn" if use_batchnorm else ""

        # Projection from encoding to bottleneck
        self.feature_shape = (init_channels, image_size[0] // 2 ** (blocks + 1), image_size[1] // 2 ** (blocks + 1))
        self.encoding2features = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_fc", nn.Linear(latent_dim, reduce(mul, self.feature_shape))),
                    ("bottleneck_elu", nn.ELU(inplace=True)),
                ]
            )
        )

        # Upsampling transposed convolution blocks
        self.features2output = nn.Sequential()
        block_in_channels = init_channels
        for idx, block_idx in enumerate(reversed(range(blocks))):
            block_out_channels = init_channels * 2 ** block_idx
            self.features2output.add_module(
                f"conv_transpose{batchnorm_desc}_elu{idx}",
                conv_transpose2x2_bn_activation(
                    in_channels=block_in_channels, out_channels=block_out_channels, bn=use_batchnorm, activation="ELU"
                ),
            )
            self.features2output.add_module(
                f"conv{batchnorm_desc}_elu{idx}",
                conv3x3_bn_activation(
                    in_channels=block_out_channels, out_channels=block_out_channels, bn=use_batchnorm, activation="ELU"
                ),
            )
            block_in_channels = block_out_channels

        self.features2output.add_module(
            f"conv_transpose{batchnorm_desc}_elu{blocks}",
            conv_transpose2x2_bn_activation(
                in_channels=block_in_channels, out_channels=block_in_channels, bn=use_batchnorm, activation="ELU"
            ),
        )

        # Classifier
        self.classifier = nn.Conv2d(block_in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            z: (N, ``latent_dim``), Encoding of the input in the latent space.

        Returns:
            (N, ``channels``, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
        """
        features = self.encoding2features(z)
        features = self.features2output(features.view((-1, *self.feature_shape)))
        out = self.classifier(features)
        return out
