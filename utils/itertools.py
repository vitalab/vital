from abc import ABC
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable


class IterableResult(Iterable, ABC):
    unit: str
    index_name: str

    def __init__(self, results_path: Path):
        self.results_path = results_path

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        pass
