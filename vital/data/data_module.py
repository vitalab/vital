import os
from abc import ABC
from argparse import ArgumentParser
from typing import Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from vital.data.config import DataParameters, Subset


class VitalDataModule(pl.LightningDataModule, ABC):
    """Top-level abstract data module from which to inherit.

    Implementations of behaviors related to data handling (e.g. data preparation) are made through this class.
    """

    def __init__(self, data_params: DataParameters, batch_size: int, num_workers: int = os.cpu_count() - 1, **kwargs):
        """Initializes class instance.

        References:
            - ``num_workers`` documentation, for more detail:
              https://lightning.ai/docs/pytorch/stable/advanced/speed.html#num-workers

        Args:
            data_params: Parameters related to the data necessary to initialize networks working with this dataset.
            batch_size: Size of batches.
            num_workers: Number of subprocesses to use for data loading. ``num_workers=0`` means that the data will be
                loaded in the main process.
        """
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets: Dict[Subset, Dataset] = {}
        self.save_hyperparameters(ignore="data_params")

    def _dataloader(self, subset: Subset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.datasets[subset],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers),
        )

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return self._dataloader(Subset.TRAIN, shuffle=True)

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return self._dataloader(Subset.VAL)

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        return self._dataloader(Subset.TEST)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:  # noqa: D102
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, required=True, help="Size of batches")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=os.cpu_count() - 1,
            help="Number of subprocesses to use for data loading. ``workers=0`` means that the data will be loaded in "
            "the main process",
        )
        return parser
