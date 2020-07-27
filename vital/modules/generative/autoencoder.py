from typing import Tuple

from torch import Tensor, nn

from vital.modules.generative.decoder import Decoder
from vital.modules.generative.encoder import Encoder
from vital.modules.layers import reparameterize


class Autoencoder(nn.Module):
    """Module making up a fully convolutional autoencoder."""

    def __init__(
        self, image_size: Tuple[int, int], channels: int, blocks: int, init_channels: int, latent_dim: int
    ):  # noqa: D205,D212,D415
        """
        Args:
            image_size: Size of the output segmentation groundtruth for each axis.
            channels: Number of channels of the image to reconstruct.
            blocks: Number of upsampling transposed convolution blocks to use.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            latent_dim: Number of dimensions in the latent space.
        """
        super().__init__()
        self.encoder = Encoder(
            image_size=image_size,
            in_channels=channels,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            image_size=image_size,
            out_channels=channels,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
        )

    def forward(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Defines the computation performed at every call.

        Args:
            y: (N, ``channels``, H, W), Input to reconstruct.

        Returns:
            - (N, ``channels``, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
            - (N, ``latent_dim``), Encoding of the input in the latent space.
        """
        z = self.encoder(y)
        return self.decoder(z), z


class VariationalAutoencoder(nn.Module):
    """Module making up a fully convolutional variational autoencoder."""

    def __init__(
        self, image_size: Tuple[int, int], channels: int, blocks: int, init_channels: int, latent_dim: int
    ):  # noqa: D205,D212,D415
        """
        Args:
            image_size: Size of the output segmentation groundtruth for each axis.
            channels: Number of channels of the image to reconstruct.
            blocks: Number of upsampling transposed convolution blocks to use.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            latent_dim: Number of dimensions in the latent space.
        """
        super().__init__()
        self.encoder = Encoder(
            image_size=image_size,
            in_channels=channels,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
            output_distribution=True,
        )
        self.decoder = Decoder(
            image_size=image_size,
            out_channels=channels,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
        )

    def forward(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Defines the computation performed at every call.

        Args:
            y: (N, ``channels``, H, W), Input to reconstruct.

        Returns:
            - (N, ``channels``, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
            - (N, ``latent_dim``), Sampled encoding of the input in the latent space.
            - (N, ``latent_dim``), Mean of the predicted distribution of the input in the latent space, used to sample
              ``z``.
            - (N, ``latent_dim``), Log variance of the predicted distribution of the input in the latent space, used to
              sample ``z``.
        """
        mu, logvar = self.encoder(y)
        z = reparameterize(mu, logvar)
        return self.decoder(z), z, mu, logvar
