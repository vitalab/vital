from abc import ABC
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Sized, TypeVar

Result = TypeVar('Result')


class IterableResult(Iterable[Result], Sized, ABC):
    """Interface to implement for iterables over systems' results to work with generic logs."""
    desc: str  # Description of the iterable unit. Used in e.g. progress bar, metrics' index header, etc.

    def __init__(self, results_path: Path):
        """
        Args:
            results_path: root path of the results over which to iterate.
        """
        self.results_path = results_path

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """Adds arguments required to configure the iterable results to logger's CLI."""
        pass
