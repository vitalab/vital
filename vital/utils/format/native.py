from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, TypeVar, Union


def apply(obj, func):
    """Applies a function recursively to all elements inside a Python collection composed of the supported types.

    References:
        - This function was inspired by a similar function from the 'poutyne' framework:
        https://github.com/GRAAL-Research/poutyne/blob/aeb78c2b26edaa30663a88522d39a187baeec9cd/poutyne/utils.py#L104-L113

    Args:
        obj: The Python object to convert.
        func: The function to apply.

    Returns:
        A new Python collection with the same structure as `obj` but where the elements have been applied the function
        `func`. Not supported types are left as reference in the new object.
    """
    if isinstance(obj, (list, tuple)):
        return type(obj)(apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k: apply(el, func) for k, el in obj.items()}
    return func(obj)


def prefix(map: Mapping[str, Any], prefix: str, exclude: Union[str, Sequence[str]] = None) -> Dict[str, Any]:
    """Prepends a prefix to the keys of a mapping with string keys.

    Args:
        map: Mapping with string keys for which to add a prefix to the keys.
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Mapping where the keys have been prepended with `prefix`.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    return {f"{prefix}{k}" if k not in exclude else k: v for k, v in map.items()}


Item = TypeVar("Item")


def squeeze(seq: Sequence[Item]) -> Union[Item, Sequence[Item]]:
    """Extracts the item from a sequence if it is the only one, otherwise leaves the sequence as is.

    Args:
        seq: Sequence to process.

    Returns:
        Single item from the sequence, or sequence itself it has more than one item.
    """
    if len(seq) == 1:
        (seq,) = seq
    return seq


def flatten(nested_list) -> List:
    """Flattens a (nested) list recursively.

    Args:
        nested_list: List to flatten.

    Returns:
        Flattened list.
    """
    return sum(map(flatten, nested_list), []) if isinstance(nested_list, list) else [nested_list]


def filter_excluded(seq: Sequence[str], to_exclude: Sequence[Union[str, Path]]) -> List[str]:
    """Filters a sequence of strings to exclude specific strings.

    Args:
        seq: Sequence of strings to filter.
        to_exclude: Individual strings, or files listing multiple strings, to exclude.

    Returns:
        List of strings based off of `seq`, without any of the values specified in `to_exclude`.
    """
    items_to_exclude = [item for item in to_exclude if isinstance(item, str)]
    for list_items_to_exclude in [item for item in to_exclude if isinstance(item, Path)]:
        with open(str(list_items_to_exclude)) as f:
            items_to_exclude.extend(f.read().splitlines())

    return [item for item in seq if item not in items_to_exclude]
