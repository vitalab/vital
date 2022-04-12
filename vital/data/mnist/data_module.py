from pathlib import Path
from typing import Callable, Optional, Union

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from vital import get_vital_home
from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule
from vital.data.mnist.mnist import MNIST


class MnistDataModule(VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the MNIST dataset."""

    def __init__(
        self,
        dataset_path: Union[str, Path] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path where to download the dataset. If `None`, defaults to using `vital`'s home directory.
            transform: Transform to apply to the input image
            target_transform: Transform to apply to the target.
            download: If `True`, download the dataset if it is not already in `dataset_path`.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(data_params=DataParameters(in_shape=(1, 28, 28), out_shape=(10,)), **kwargs)

        dataset_path = dataset_path or get_vital_home()
        transform = transform or ToTensor()

        self._download = download
        self._dataset_kwargs = {
            "root": Path(dataset_path),
            "transform": transform,
            "target_transform": target_transform,
        }

    def prepare_data(self):  # noqa: D102
        # If `self.download` is `True`, this will download the dataset on each node
        # Otherwise, it does nothing
        MNIST(**self._dataset_kwargs, train=True, download=self._download)
        MNIST(**self._dataset_kwargs, train=False, download=self._download)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        if stage == "fit":
            self._dataset[Subset.TRAIN] = MNIST(**self._dataset_kwargs, train=True)
            self._dataset[Subset.VAL] = MNIST(**self._dataset_kwargs, train=False)
        if stage == "test":
            self._dataset[Subset.TEST] = MNIST(**self._dataset_kwargs, train=False)

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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
