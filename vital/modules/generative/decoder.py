from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Tuple

from torch import Tensor, nn

from vital.modules.layers import (
    conv2d_activation,
    conv2d_bn_activation,
    convtranspose2d_activation,
    convtranspose2d_bn_activation,
)


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
        activation: str = "ELU",
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
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
        """
        super().__init__()
        if use_batchnorm:
            conv_block = conv2d_bn_activation
            convtranspose_block = convtranspose2d_bn_activation
            batchnorm_desc = "_bn"
        else:
            conv_block = conv2d_activation
            convtranspose_block = convtranspose2d_activation
            batchnorm_desc = ""

        # Projection from encoding to bottleneck
        self.feature_shape = (init_channels, image_size[0] // 2 ** (blocks + 1), image_size[1] // 2 ** (blocks + 1))
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
        for idx, block_idx in enumerate(reversed(range(blocks))):
            block_out_channels = init_channels * 2 ** block_idx
            self.features2output.add_module(
                f"conv_transpose{batchnorm_desc}_{activation.lower()}{idx}",
                convtranspose_block(
                    in_channels=block_in_channels, out_channels=block_out_channels, activation=activation
                ),
            )
            self.features2output.add_module(
                f"conv{batchnorm_desc}_{activation.lower()}{idx}",
                conv_block(in_channels=block_out_channels, out_channels=block_out_channels, activation=activation),
            )
            block_in_channels = block_out_channels

        self.features2output.add_module(
            f"conv_transpose{batchnorm_desc}_{activation.lower()}{blocks}",
            convtranspose_block(in_channels=block_in_channels, out_channels=block_in_channels, activation=activation),
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
