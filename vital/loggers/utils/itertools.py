from abc import ABC
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Sized, TypeVar

Result = TypeVar("Result")


class IterableResult(Iterable[Result], Sized, ABC):
    """Interface describing how to iterate over results in a way that can be leveraged by the `Logger` API."""

    desc: str  #: Description of the iterable unit. Used in e.g. progress bar, metrics' index header, etc.

    def __init__(self, results_path: Path):  # noqa: D205,D212,D415
        """
        Args:
            results_path: Path of the results over which to iterate. E.g. an HDF5 dataset file, the root directory of a
                hierarchy of image files, etc.
        """
        self.results_path = results_path

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments required to configure the iterable results to logger's CLI.

        Args:
            parser: Parser object for which to add arguments for filtering results on which to iterate.

        Returns:
            Parser object with support for arguments for filtering results on which to iterate.
        """
        return parser
