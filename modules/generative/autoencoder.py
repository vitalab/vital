import torch
import torch.nn as nn

from vital.modules.generative.decoder import Decoder
from vital.modules.generative.encoder import Encoder


class Autoencoder(nn.Module):

    def __init__(self, image_size: (int, int),
                 channels: int,
                 blocks: int,
                 feature_maps: int,
                 code_length: int):
        """ Module making up a fully convolutional autoencoder.

        Args:
            image_size: size of the output segmentation groundtruth for each axis.
            channels: number of channels of the image to reconstruct.
            blocks: number of upsampling transposed convolution blocks to use.
            feature_maps: factor used to compute the number of feature maps for the convolution layers.
            code_length: number of dimensions in the latent space.
        """
        super().__init__()
        self.encoder = Encoder(image_size=image_size,
                               channels=channels,
                               blocks=blocks,
                               feature_maps=feature_maps,
                               code_length=code_length)
        self.decoder = Decoder(image_size=image_size,
                               channels=channels,
                               blocks=blocks,
                               feature_maps=feature_maps,
                               code_length=code_length)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def predict(self, x):
        return self(x)


class VariationalAutoencoder(nn.Module):

    def __init__(self, image_size: (int, int),
                 channels: int,
                 blocks: int,
                 feature_maps: int,
                 code_length: int):
        """ Module making up a fully convolutional variational autoencoder.

        Args:
            image_size: size of the output segmentation groundtruth for each axis.
            channels: number of channels of the image to reconstruct.
            blocks: number of upsampling transposed convolution blocks to use.
            feature_maps: factor used to compute the number of feature maps for the convolution layers.
            code_length: number of dimensions in the latent space.
        """
        super().__init__()
        self.encoder = Encoder(image_size=image_size,
                               channels=channels,
                               blocks=blocks,
                               feature_maps=feature_maps,
                               code_length=code_length,
                               output_distribution=True)
        self.decoder = Decoder(image_size=image_size,
                               channels=channels,
                               blocks=blocks,
                               feature_maps=feature_maps,
                               code_length=code_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z, mu, logvar

    def predict(self, x):
        z, _ = self.encoder(x)
        return self.decoder(z), z
