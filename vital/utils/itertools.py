import itertools
import typing
from abc import ABC
from argparse import ArgumentParser
from typing import Dict, List, Mapping, Sequence, Tuple, TypeVar, Union

K = TypeVar("K")
V = TypeVar("V")
Item = TypeVar("Item")


class Collection(typing.Collection[Item], ABC):
    """Wrapper around the native `Collection` that enforces functionalities leveraged by other `vital` APIs."""

    desc: str  # Description of an item from the collection. Used in e.g. logging messages, progress bar, etc.

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments required to configure the collection to an argument parser.

        Args:
            parser: Parser object for which to add arguments for configuring the collection.

        Returns:
            Parser object with support for arguments for configuring the collection.
        """
        return parser

    def __contains__(self, item):  # noqa: D105
        # TODO: Fix collection inheritance to not require this explicit
        # Normally, user-defined `Collection` types w/o a defined `__contains__` method should immediately fall back to
        # using `__iter__`. But for some reason, if `__contains__` is not implemented here, any class derived from our
        # `Collection` type will raise a `TypeError` at instantiation reading:
        # "Can't instantiate abstract class {CLASS_NAME} with abstract method __contains__"
        # Therefore, the current implementation of `__contains__` is required to avoid this bug.
        return super().__contains__(item)


def dict_values_cartesian_product(matrix: Mapping[K, List[V]]) -> List[Dict[K, V]]:
    """Lists all elements of the cartesian product between the mapping's values.

    Args:
        matrix: Mapping between keys and lists of values.

    Returns:
        List of all the elements in the cartesian product between the mapping's values.

    Example:
        >>> matrix = {"a": [1, 2], "b": [3, 4]}
        >>> dict_values_cartesian_product(matrix)
        [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    """
    return [dict(zip(matrix.keys(), matrix_elem_values)) for matrix_elem_values in itertools.product(*matrix.values())]


def listify_dict_values(elem_mapping: Mapping[K, Union[List, Tuple, V]]) -> Dict[K, Sequence[V]]:
    """Ensures the values in a mapping are sequences and don't contain lone elements.

    Args:
        elem_mapping: Mapping between keys and single values or sequence of values.

    Returns:
        Mapping between keys and sequences of values.

    Example:
        >>> elem_mapping = {"a": 1, "bc": [2, 3]}
        >>> listify_dict_values(elem_mapping)
        {"a": [1], "bc": [2, 3]}
    """
    return {k: v if isinstance(v, (list, tuple)) else [v] for k, v in elem_mapping.items()}
