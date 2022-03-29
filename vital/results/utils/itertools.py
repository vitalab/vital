from abc import ABC
from argparse import ArgumentParser
from typing import Iterable, Sized, TypeVar

Result = TypeVar("Result")


class IterableResult(Iterable[Result], Sized, ABC):
    """Interface describing how to iterate over results in a way that can be leveraged by the `Logger` API."""

    desc: str  #: Description of the iterable unit. Used in e.g. progress bar, metrics' index header, etc.

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments required to configure the iterable results to logger's CLI.

        Args:
            parser: Parser object for which to add arguments for filtering results on which to iterate.

        Returns:
            Parser object with support for arguments for filtering results on which to iterate.
        """
        return parser
