from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from vital.models.classification.mlp import MLP
from vital.models.generative.decoder import Decoder1d, Decoder2d
from vital.models.generative.encoder import Encoder1d, Encoder2d
from vital.models.layers import reparameterize


class MLPAutoencoder(nn.Module):
    """Module making up a fully connected N-D autoencoder."""

    # Tags used as keys in the dict returned by the forward pass
    x_hat_tag: str = "x_hat"
    z_tag: str = "z"

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        latent_dim: int,
        encoder_hidden: Optional[Tuple[int, ...]] = None,
        decoder_hidden: Optional[Tuple[int, ...]] = None,
        activation: str = "ReLU",
        encoder_activation: Optional[str] = None,
        dropout: float = 0,
    ):
        """Initializes class instance.

        Args:
            input_shape: Shape of the inputs to reconstruct. If this does not specify a single dim, the input will be
                automatically flattened at the start of the forward pass.
            latent_dim: Number of dimensions in the latent space.
            encoder_hidden: Number of neurons at each hidden layer of the encoder. If None or empty, the MLP will
                correspond to a linear model between the input and encoding.
            decoder_hidden: Number of neurons at each hidden layer of the decoder. If None, the decoder's hidden layers
                will automatically be determined to mirror the encoder. If empty, the decoder will correspond to a
                linear model between the encoding and reconstruction.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                encoder and decoder's hidden layers.
            encoder_activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use at the
                output of the encoder.
            dropout: Rate for dropout layers.
        """
        super().__init__()

        if encoder_hidden is None:
            encoder_hidden = []
        if decoder_hidden is None:
            decoder_hidden = encoder_hidden[::-1]

        self.encoder = MLP(
            input_shape=input_shape,
            output_shape=(latent_dim,),
            hidden=encoder_hidden,
            activation=activation,
            output_activation=encoder_activation,
            dropout=dropout,
        )
        self.decoder = MLP(
            input_shape=(latent_dim,),
            output_shape=input_shape,
            hidden=decoder_hidden,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ...), Input to reconstruct.

        Returns:
            Dict with values:
            - (N, ...), Reconstructed input.
            - (N, ``latent_dim``), Encoding of the input in the latent space.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return {self.x_hat_tag: x_hat, self.z_tag: z}


class Autoencoder2d(nn.Module):
    """Module making up a fully convolutional 2D autoencoder."""

    # Tags used as keys in the dict returned by the forward pass
    reconstruction_tag: str = "x_hat"
    encoding_tag: str = "z"

    #:  Whether the encoder has a second head to output the ``logvar`` along with the default ``mu`` head
    output_distribution: bool = False

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        blocks: int,
        init_channels: int,
        latent_dim: int,
        input_latent_dim: int = 0,
        activation: str = "ELU",
        use_batchnorm: bool = True,
    ):
        """Initializes class instance.

        Args:
            input_shape: (C, H, W), Shape of the inputs to reconstruct.
            blocks: Number of upsampling transposed convolution blocks to use.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            latent_dim: Number of dimensions in the latent space.
            input_latent_dim: Number of dimensions to add to the latent space prior to decoding. These are not predicted
                by the encoder, but come from auxiliary inputs to the network.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
            use_batchnorm: Whether to use batch normalization between the convolution and activation layers in the
                convolutional blocks.
        """
        super().__init__()
        self.encoder = Encoder2d(
            input_shape=input_shape,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
            output_distribution=self.output_distribution,
        )
        self.decoder = Decoder2d(
            output_shape=input_shape,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim + input_latent_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: Tensor, subencodings: Sequence[Tensor] = None) -> Dict[str, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, C, H, W), Input to reconstruct.
            subencodings: (N, ?) tensors representing subspaces of the latent space, to be concatenated to the subspace
                predicted by the encoder to give the complete latent space vectors.
                When summed together, the second dimensions of these tensors should equal ``input_latent_dim``.

        Returns:
            Dict with values:
            - (N, C, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
            - (N, ``latent_dim``), Encoding of the input in the latent space.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z if subencodings is None else torch.cat((*subencodings, z), dim=1))
        return {self.reconstruction_tag: x_hat, self.encoding_tag: z}


class VariationalAutoencoder2d(Autoencoder2d):
    """Module making up a fully convolutional variational 2D autoencoder."""

    # Tags used as keys in the dict returned by the forward pass
    distr_mean_tag: str = "mu"
    distr_logvar_tag: str = "logvar"

    output_distribution = True

    def forward(self, x: Tensor, subencodings: Sequence[Tensor] = None) -> Dict[str, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, C, H, W), Input to reconstruct.
            subencodings: (N, ?) tensors representing subspaces of the latent space, to be concatenated to the subspace
                predicted by the encoder to give the complete latent space vectors.
                When summed together, the second dimensions of these tensors should equal ``input_latent_dim``.

        Returns:
            Dict with values:
            - (N, C, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
            - (N, ``latent_dim``), Sampled encoding of the input in the latent space.
            - (N, ``latent_dim``), Mean of the predicted distribution of the input in the latent space, used to sample
              ``z``.
            - (N, ``latent_dim``), Log variance of the predicted distribution of the input in the latent space, used to
              sample ``z``.
        """
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z if subencodings is None else torch.cat((*subencodings, z), dim=1))
        return {
            self.reconstruction_tag: x_hat,
            self.encoding_tag: z,
            self.distr_mean_tag: mu,
            self.distr_logvar_tag: logvar,
        }


class Autoencoder1d(nn.Module):
    """Module making up a fully convolutional 1D autoencoder."""

    # Tags used as keys in the dict returned by the forward pass
    x_hat_tag: str = "x_hat"
    z_tag: str = "z"

    #:  Whether the encoder has a second head to output the ``logvar`` along with the default ``mu`` head
    output_distribution: bool = False

    def __init__(
        self, input_shape: Tuple[int, int], blocks: int, init_channels: int, latent_dim: int, activation: str = "ELU"
    ):
        """Initializes class instance.

        Args:
            input_shape: (C, L), Shape of the inputs to reconstruct.
            blocks: Number of downsampling convolution blocks to use.
            init_channels: Number of output channels from the first layer, used to compute the number of channels in
                following layers.
            latent_dim: Number of dimensions in the latent space.
            activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use across the
                network.
        """
        super().__init__()
        self.encoder = Encoder1d(
            input_shape=input_shape,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
            activation=activation,
            output_distribution=self.output_distribution,
        )
        self.decoder = Decoder1d(
            output_shape=input_shape,
            blocks=blocks,
            init_channels=init_channels,
            latent_dim=latent_dim,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, C, L), Input to reconstruct.

        Returns:
            Dict with values:
            - (N, C, L), Reconstructed input.
            - (N, ``latent_dim``), Encoding of the input in the latent space.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return {self.x_hat_tag: x_hat, self.z_tag: z}


class VariationalAutoencoder1d(Autoencoder1d):
    """Module making up a fully convolutional variational 1D autoencoder."""

    # Tags used as keys in the dict returned by the forward pass
    mu_tag: str = "mu"
    logvar_tag: str = "logvar"

    output_distribution = True

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, C, L), Input to reconstruct.

        Returns:
            Dict with values:
            - (N, C, L), Reconstructed input.
            - (N, ``latent_dim``), Encoding of the input in the latent space.
            - (N, ``latent_dim``), Mean of the predicted distribution of the input in the latent space.
            - (N, ``latent_dim``), Log variance of the predicted distribution of the input in the latent space.
        """
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return {self.x_hat_tag: x_hat, self.z_tag: z, self.mu_tag: mu, self.logvar_tag: logvar}
