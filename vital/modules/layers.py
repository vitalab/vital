from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple

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


def _add_batchnorm(
    dimensionality: Literal["1d", "2d", "3d"],
) -> Callable[[Callable[..., Sequence[Tuple[str, nn.Module]]]], Callable[..., Sequence[Tuple[str, nn.Module]]]]:
    """Decorator for functions that return a sequence of layers, to add a batchnorm layer at the end.

    Args:
        dimensionality: Literal indicating the type of batchnorm to add to the list of layers.

    Returns:
        Function were a batchnorm layer is appended to the base sequence of layers.
    """
    if dimensionality == "1d":
        feature_attr = "out_features"
    else:
        feature_attr = "out_channels"

    def _add_batchnorm_decorator(
        fn: Callable[..., Sequence[Tuple[str, nn.Module]]]
    ) -> Callable[..., Sequence[Tuple[str, nn.Module]]]:
        @wraps(fn)
        def layer_with_batchnorm(*args, bn: bool = True, bn_kwargs: Dict[str, Any] = None, **kwargs):
            layer_modules = fn(*args, **kwargs)
            if bn:
                layer_modules = list(layer_modules)
                num_features = getattr(layer_modules[-1][1], feature_attr)
                if bn_kwargs is None:
                    bn_kwargs = {}
                layer_modules.append(
                    (
                        f"batchnorm{dimensionality}",
                        _get_nn_module(f"BatchNorm{dimensionality}", num_features, **bn_kwargs),
                    )
                )
            return layer_modules

        return layer_with_batchnorm

    return _add_batchnorm_decorator


def _add_activation(
    fn: Callable[..., Sequence[Tuple[str, nn.Module]]]
) -> Callable[..., Sequence[Tuple[str, nn.Module]]]:
    """Decorator for functions that return a sequence of layers, to add an activation layer at the end.

    Args:
        fn: Function that returns a sequence of layers' i) name and ii) object.

    Returns:
        Function were an activation layer is appended to the base sequence of layers.
    """

    @wraps(fn)
    def layer_with_activation(*args, activation: str = "ReLU", activation_kwargs: Dict[str, Any] = None, **kwargs):
        layer_modules = fn(*args, **kwargs)
        if activation:
            layer_modules = list(layer_modules)
            if activation_kwargs is None:
                activation_kwargs = {}
            layer_modules.append((activation, _get_nn_module(activation, **activation_kwargs)))
        return list(layer_modules)

    return layer_with_activation


def _build_sequential(fn: Callable[..., Sequence[Tuple[str, nn.Module]]]) -> Callable[..., nn.Sequential]:
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


@_build_sequential
@_add_activation
@_add_batchnorm("2d")
def conv_transpose2x2_bn_activation(
    in_channels: int, out_channels: int, stride: int = 2, padding: int = 0
) -> List[Tuple[str, nn.Module]]:
    """2x2 transpose convolution with padding followed by optionals batch normalization and activation."""
    return [
        ("conv_transpose", nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=padding))
    ]


@_build_sequential
@_add_activation
@_add_batchnorm("2d")
def conv3x3_bn_activation(
    in_channels: int, out_channels: int, stride: int = 1, padding: int = 1
) -> List[Tuple[str, nn.Module]]:
    """3x3 convolution with padding followed by optionals batch normalization and activation."""
    return [("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding))]


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
