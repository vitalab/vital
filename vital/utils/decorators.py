from functools import wraps
from typing import Any, Callable, Dict, Mapping, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from vital.utils.format.native import Item
from vital.utils.format.native import prefix as prefix_fn
from vital.utils.format.native import squeeze as squeeze_fn


def prefix(
    prefix: str, exclude: Union[str, Sequence[str]] = None
) -> Callable[[Callable[..., Mapping[str, Any]]], Callable[..., Dict[str, Any]]]:
    """Decorator for functions that return a mapping with string keys, to add a prefix to the keys.

    Args:
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Function where the keys of the mapping returned have been prepended with `prefix`, except for the keys listed in
        `exclude`, that are left as-is.
    """

    def prefix_decorator(fn: Callable[..., Mapping[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @wraps(fn)
        def prefix_wrapper(self, *args, **kwargs):
            return prefix_fn(fn(self, *args, **kwargs), prefix, exclude=exclude)

        return prefix_wrapper

    return prefix_decorator


def squeeze(fn: Callable[..., Sequence[Item]]) -> Callable[..., Union[Item, Sequence[Item]]]:
    """Decorator for functions that return sequences of possibly one item, where we would want the lone item directly.

    Args:
        fn: Function that returns a sequence.

    Returns:
        Function that returns a single item from the sequence, or the sequence itself it has more than one item.
    """

    @wraps(fn)
    def unpack_single_item_return(*args, **kwargs):
        return squeeze_fn(fn(*args, **kwargs))

    return unpack_single_item_return


def auto_cast_data(func: Callable) -> Callable:
    """Decorator to allow functions relying on numpy arrays to accept other input data types.

    Args:
        func: Function for which to automatically convert the first argument to a numpy array.

    Returns:
        Function that accepts input data types other than numpy arrays by converting between them and numpy arrays.
    """
    cast_types = [Tensor]
    dtypes = [np.ndarray, *cast_types]

    @wraps(func)
    def _call_func_with_cast_data(data, *args, **kwargs):
        dtype = type(data)
        if dtype not in dtypes:
            raise ValueError(
                f"Decorator 'auto_cast_data' used by function '{func.__name__}' does not support casting inputs of "
                f"type '{cast_types}' to numpy arrays. Either provide the implementation for casting to numpy arrays "
                f"from '{cast_types}' in 'auto_cast_data' decorator, or manually convert the input of '{func.__name__}'"
                f"to one of the following supported types: {dtypes}."
            )
        if dtype == Tensor:
            data_device = data.device
            data = data.detach().cpu().numpy()
        result = func(data, *args, **kwargs)
        if dtype == Tensor:
            result = torch.tensor(result, device=data_device)
        return result

    return _call_func_with_cast_data
