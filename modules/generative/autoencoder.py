from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from vital.modules.generative.decoder import Decoder
from vital.modules.generative.encoder import Encoder


class Autoencoder(nn.Module):
    """Module making up a fully convolutional autoencoder."""

    def __init__(self, image_size: Tuple[int, int],
                 channels: int,
                 blocks: int,
                 init_channels: int,
                 code_length: int):
        """
        Args:
            image_size: size of the output segmentation groundtruth for each axis.
            channels: number of channels of the image to reconstruct.
            blocks: number of upsampling transposed convolution blocks to use.
            init_channels: number of output feature maps from the first layer, used to compute the number of feature
                           maps in following layers.
            code_length: number of dimensions in the latent space.
        """
        super().__init__()
        self.encoder = Encoder(image_size=image_size,
                               in_channels=channels,
                               blocks=blocks,
                               init_channels=init_channels,
                               code_length=code_length)
        self.decoder = Decoder(image_size=image_size,
                               out_channels=channels,
                               blocks=blocks,
                               init_channels=init_channels,
                               code_length=code_length)

    def forward(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """ Defines the computation performed at every call.

        Args:
            y: (N, ``channels``, H, W), input to reconstruct.

        Returns:
            y_hat: (N, ``channels``, H, W), raw, unnormalized scores for each class in the input's reconstruction.
            z: (N, ``code_length``), encoding of the input in the latent space.
        """
        z = self.encoder(y)
        return self.decoder(z), z

    def predict(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """ Performs test-time inference on the input.

        Args:
            y: (N, ``channels``, H, W), input to reconstruct.

        Returns:
            y_hat: (N, ``channels``, H, W), raw, unnormalized scores for each class in the input's reconstruction.
            z: (N, ``code_length``), encoding of the input in the latent space.
        """
        return self(y)


class VariationalAutoencoder(nn.Module):
    """Module making up a fully convolutional variational autoencoder."""

    def __init__(self, image_size: Tuple[int, int],
                 channels: int,
                 blocks: int,
                 init_channels: int,
                 code_length: int):
        """
        Args:
            image_size: size of the output segmentation groundtruth for each axis.
            channels: number of channels of the image to reconstruct.
            blocks: number of upsampling transposed convolution blocks to use.
            init_channels: number of output feature maps from the first layer, used to compute the number of feature
                           maps in following layers.
            code_length: number of dimensions in the latent space.
        """
        super().__init__()
        self.encoder = Encoder(image_size=image_size,
                               in_channels=channels,
                               blocks=blocks,
                               init_channels=init_channels,
                               code_length=code_length,
                               output_distribution=True)
        self.decoder = Decoder(image_size=image_size,
                               out_channels=channels,
                               blocks=blocks,
                               init_channels=init_channels,
                               code_length=code_length)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """

        Args:
            mu: (N, ``code_length``), mean of the predicted distribution of the input in the latent space.
            logvar: (N, ``code_length``), log variance of the predicted distribution of the input in the latent space.

        Returns:
            z: (N, ``code_length``), sampled encoding of the input in the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ Defines the computation performed at every call.

        Args:
            y: (N, ``channels``, H, W), input to reconstruct.

        Returns:
            y_hat: (N, ``channels``, H, W), raw, unnormalized scores for each class in the input's reconstruction.
            z: (N, ``code_length``), sampled encoding of the input in the latent space.
            mu: (N, ``code_length``), mean of the predicted distribution of the input in the latent space,
                used to sample ``z``.
            logvar: (N, ``code_length``), log variance of the predicted distribution of the input in the latent space,
                used to sample ``z``.
        """
        mu, logvar = self.encoder(y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z, mu, logvar

    def predict(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """ Performs test-time inference on the input.

        Args:
            y: (N, ``channels``, H, W), input to reconstruct.

        Returns:
            y_hat: (N, ``channels``, H, W), raw, unnormalized scores for each class in the input's reconstruction.
            z: (N, ``code_length``), deterministic encoding of the input in the latent space.
        """
        z, _ = self.encoder(y)
        return self.decoder(z), z
