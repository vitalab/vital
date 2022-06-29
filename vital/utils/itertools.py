import itertools
import typing
from abc import ABC
from argparse import ArgumentParser
from typing import Dict, List, Mapping, Sequence, Sized, Tuple, TypeVar, Union

K = TypeVar("K")
V = TypeVar("V")
Item = TypeVar("Item")


class Iterable(typing.Iterable[Item], Sized, ABC):
    """Wrapper around the native `Iterable` that enforces functionalities leveraged by other `vital` APIs."""

    desc: str  # Description of an iterable item. Used in e.g. logging messages, progress bar, etc.

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments required to configure the iterable to an argument parser.

        Args:
            parser: Parser object for which to add arguments for configuring iterable.

        Returns:
            Parser object with support for arguments for configuring iterable.
        """
        return parser


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
