from functools import wraps
from typing import Any, Callable, Dict, Mapping, Sequence, TypeVar, Union


def prefix(prefix: str, exclude: Union[str, Sequence[str]] = None):
    """Decorator for functions that return a mapping of keys and values, where we would want to add a common prefix to
    the keys used.

    Args:
        prefix: prefix to add to the current keys in the mapping. To note that the prefix will be separated by an
                underscore (``_``) from the current keys.
        exclude: keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        functions where the keys of the mapping returned have been prepended with ``prefix``, except for the keys listed
        in ``exclude``, that are left as-is.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    def prefix_decorator(fn: Callable[..., Mapping[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @wraps(fn)
        def prefix_wrapper(self, *args, **kwargs):
            return {f"{prefix}_{k}" if k not in exclude else k: v for k, v in fn(self, *args, **kwargs).items()}

        return prefix_wrapper

    return prefix_decorator


Item = TypeVar("Item")


def squeeze(fn: Callable[..., Sequence[Item]]) -> Callable[..., Union[Item, Sequence[Item]]]:
    """Decorator for functions that return sequences of possibly one item, where we would want the lone item directly.

    Args:
        fn: a function that returns a sequence.

    Returns:
        a function that returns a sequence, or directly an item if the sequence only contains one item.
    """

    @wraps(fn)
    def unpack_single_item_return(*args, **kwargs):
        out = fn(*args, **kwargs)
        if len(out) == 1:
            (out,) = out
        return out

    return unpack_single_item_return
