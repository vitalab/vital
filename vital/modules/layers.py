from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn


def _get_nn_module(module: str, *module_args, **module_kwargs) -> nn.Module:
    """Instantiates an ``nn.Module`` with the requested parameters.

    Args:
        module: Name of the ``nn.Module`` to instantiate.
        *module_args: Positional arguments to pass to the ``nn.Module``'s constructor.
        **module_kwargs: Keyword arguments to pass to the ``nn.Module``'s constructor.

    Returns:
        Instance of the ``nn.Module``.
    """
    return getattr(nn, module)(*module_args, **module_kwargs)


def _sequential(fn: Callable[..., Sequence[Tuple[str, nn.Module]]]) -> Callable[..., nn.Sequential]:
    """Decorator for functions that return a sequence of layers, that we would want as a single sequential module.

    Args:
        fn: Function that returns a sequence of layers' i) name and ii) object.

    Returns:
        Function were the requested sequence of layers is bundled as a single sequential module.
    """

    @wraps(fn)
    def layer_as_sequential(*args, **kwargs):
        return nn.Sequential(OrderedDict(fn(*args, **kwargs)))

    return layer_as_sequential


@_sequential
def convtranspose2d_activation(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
    conv_kwargs: Dict[str, Any] = None,
    activation: str = "ReLU",
    activation_kwargs: Dict[str, Any] = None,
) -> List[Tuple[str, nn.Module]]:
    """2D strided transpose convolution followed by activation."""
    if conv_kwargs is None:
        conv_kwargs = {}
    if activation_kwargs is None:
        activation_kwargs = {}
    return [
        (
            "conv_transpose",
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, **conv_kwargs),
        ),
        (activation, _get_nn_module(activation, **activation_kwargs)),
    ]


@_sequential
def convtranspose2d_bn_activation(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
    conv_kwargs: Dict[str, Any] = None,
    bn_kwargs: Dict[str, Any] = None,
    activation: str = "ReLU",
    activation_kwargs: Dict[str, Any] = None,
) -> List[Tuple[str, nn.Module]]:
    """2D strided transpose convolution followed by batch normalization and activation."""
    if conv_kwargs is None:
        conv_kwargs = {}
    if bn_kwargs is None:
        bn_kwargs = {}
    if activation_kwargs is None:
        activation_kwargs = {}
    return [
        (
            "conv_transpose",
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False, **conv_kwargs
            ),
        ),
        ("bn", nn.BatchNorm2d(out_channels, **bn_kwargs)),
        (activation, _get_nn_module(activation, **activation_kwargs)),
    ]


@_sequential
def conv2d_activation(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 1,
    conv_kwargs: Dict[str, Any] = None,
    activation: str = "ReLU",
    activation_kwargs: Dict[str, Any] = None,
) -> List[Tuple[str, nn.Module]]:
    """2D convolution with padding followed by activation."""
    if conv_kwargs is None:
        conv_kwargs = {}
    if activation_kwargs is None:
        activation_kwargs = {}
    return [
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, **conv_kwargs)),
        (activation, _get_nn_module(activation, **activation_kwargs)),
    ]


@_sequential
def conv2d_bn_activation(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 1,
    conv_kwargs: Dict[str, Any] = None,
    bn_kwargs: Dict[str, Any] = None,
    activation: str = "ReLU",
    activation_kwargs: Dict[str, Any] = None,
) -> List[Tuple[str, nn.Module]]:
    """2D convolution with padding followed by batch normalization and activation."""
    if conv_kwargs is None:
        conv_kwargs = {}
    if bn_kwargs is None:
        bn_kwargs = {}
    if activation_kwargs is None:
        activation_kwargs = {}
    return [
        (
            "conv",
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, **conv_kwargs),
        ),
        ("bn", nn.BatchNorm2d(out_channels, **bn_kwargs)),
        (activation, _get_nn_module(activation, **activation_kwargs)),
    ]


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
