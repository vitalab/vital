from pathlib import Path
from typing import Literal, Union

from torch.utils.data import DataLoader

from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule
from vital.data.HMC_QU.config import Label, image_size, in_channels
from vital.data.HMC_QU.dataset import HMC_QU


class HMC_QUDataModule(VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the ACDC dataset."""

    def __init__(self, dataset_path: Union[str, Path], predict_on_test: bool = True, **kwargs):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            predict_on_test: If True, add predict=True to dataset for test set to get full patients at each batch.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(
            data_params=DataParameters(
                in_shape=(in_channels, image_size, image_size),
                out_shape=(len(Label), image_size, image_size),
                labels=tuple(Label),
            ),
            **kwargs,
        )

        self._dataset_kwargs = {"path": Path(dataset_path)}
        self.predict_on_test = predict_on_test

    def setup(self, stage: Literal["fit", "test"]) -> None:  # noqa: D102
        if stage == "fit":
            self._dataset[Subset.TRAIN] = HMC_QU(image_set=Subset.TRAIN, **self._dataset_kwargs)
            self._dataset[Subset.VAL] = HMC_QU(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == "test":
            self._dataset[Subset.TEST] = HMC_QU(
                image_set=Subset.TEST, predict=self.predict_on_test, **self._dataset_kwargs
            )

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.dataset(subset=Subset.TRAIN),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.dataset(subset=Subset.VAL),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.dataset(subset=Subset.TEST),
            # batch_size=None returns one full patient at each step.
            batch_size=None if self.predict_on_test else self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
