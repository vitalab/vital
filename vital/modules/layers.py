from collections import OrderedDict

import torch
from torch import Tensor, nn


def _get_module(module: str, **module_params) -> nn.Module:
    """Instantiates an ``nn.Module`` with the requested parameters.

    Args:
        module: Name of the ``nn.Module`` to instantiate.
        **module_params: Parameters to pass to the ``nn.Module``'s constructor.

    Returns:
        Instance of the ``nn.Module``.
    """
    return getattr(nn, module)(**module_params)


def conv_transpose2x2_activation(
    in_channels: int,
    out_channels: int,
    stride: int = 2,
    padding: int = 0,
    activation: str = "ReLU",
    **activation_kwargs,
) -> nn.Sequential:
    """2x2 transpose convolution with padding followed by activation."""
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv_transpose",
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=padding),
                ),
                (activation, _get_module(activation, **activation_kwargs)),
            ]
        )
    )


def conv3x3_activation(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 1,
    activation: str = "ReLU",
    **activation_kwargs,
) -> nn.Sequential:
    """3x3 convolution with padding followed by activation."""
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)),
                (activation, _get_module(activation, **activation_kwargs)),
            ]
        )
    )


def conv3x3_bn_activation(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 1,
    activation: str = "ReLU",
    **activation_kwargs,
) -> nn.Sequential:
    """3x3 convolution with padding followed by batch normalization and activation."""
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)),
                ("bn", nn.BatchNorm2d(out_channels)),
                (activation, _get_module(activation, **activation_kwargs)),
            ]
        )
    )


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Samples item from a distribution in a way that allows backpropagation to flow through.

    Args:
        mu: (N, M), Mean of the distribution.
        logvar: (N, M), Log variance of the distribution.

    Returns:
        (N, M), Item sampled from the distribution.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
