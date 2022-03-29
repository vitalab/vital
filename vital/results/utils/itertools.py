from abc import ABC
from argparse import ArgumentParser
from typing import Iterable, Sized, TypeVar

Result = TypeVar("Result")


class IterableResult(Iterable[Result], Sized, ABC):
    """Interface describing how to iterate over results in a way that can be leveraged by the `ResultsProcessor` API."""

    desc: str  #: Description of the iterable unit. Used in e.g. progress bar, metrics' index header, etc.

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments required to configure the iterable results to an argument parser.

        Args:
            parser: Parser object for which to add arguments for configuring results on which to iterate.

        Returns:
            Parser object with support for arguments for configuring results on which to iterate.
        """
        return parser
