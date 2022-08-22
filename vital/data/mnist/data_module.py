from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import Dataset, random_split
from torchvision import transforms as transform_lib

from vital import get_vital_home
from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule
from vital.data.mnist.mnist import MNIST


class MnistDataModule(VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the MNIST dataset."""

    def __init__(
        self,
        dataset_path: Union[str, Path] = None,
        val_split: Union[int, float] = 0.2,
        normalize: bool = False,
        seed: int = 0,
        transforms: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path where to download the dataset. If `None`, defaults to using `vital`'s home directory.
            val_split: Percent (float) or number (int) of samples to use for the validation split.
            normalize: If `True` applies image normalization.
            seed: Random seed to be used for train/val/test splits.
            transforms: Transforms to apply to the input image
            download: If `True`, download the dataset if it is not already in `dataset_path`.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(data_params=DataParameters(in_shape=(1, 28, 28), out_shape=(10,)), **kwargs)
        self._root = str(dataset_path or get_vital_home())
        self._transforms = transforms
        self._val_split = val_split
        self._normalize = normalize
        self._seed = seed
        self._download = download

    def prepare_data(self):  # noqa: D102
        # If `self.download` is `True`, this will download the dataset on each node
        # Otherwise, it does nothing
        MNIST(root=self._root, train=True, download=self._download)
        MNIST(root=self._root, train=False, download=self._download)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        if stage == TrainerFn.FITTING:
            # Initialize one dataset for train/val split
            transforms = self.default_transforms() if self._transforms is None else self._transforms
            dataset_train = MNIST(root=self._root, transform=transforms, train=True)
            # Split
            self.datasets[Subset.TRAIN] = self._split_dataset(dataset_train)
            self.datasets[Subset.VAL] = self._split_dataset(dataset_train, train=False)
        if stage == TrainerFn.TESTING:
            self.datasets[Subset.TEST] = MNIST(root=self._root, transform=self.default_transforms(), train=False)

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set.

        Notes:
            - Adapted from the method with the same name in the generic VisionDataModule from Lightning Bolts:
              https://github.com/PyTorchLightning/lightning-bolts/blob/2415b49a2b405693cd499e09162c89f807abbdc4/pl_bolts/datamodules/vision_datamodule.py#L83-L91
        """
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(self._seed))

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set.

        Notes:
            - Adapted from the method with the same name in the generic VisionDataModule from Lightning Bolts:
              https://github.com/PyTorchLightning/lightning-bolts/blob/2415b49a2b405693cd499e09162c89f807abbdc4/pl_bolts/datamodules/vision_datamodule.py#L93-L105
        """
        if isinstance(self._val_split, int):
            train_len = len_dataset - self._val_split
            splits = [train_len, self._val_split]
        elif isinstance(self._val_split, float):
            val_len = int(self._val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self._val_split)}")

        return splits

    def default_transforms(self) -> Callable:
        """Configures the default normalization to apply to the dataset.

        Notes:
            - Adapted from the method with the same name in the MNISTDataModule from Lightning Bolts:
              https://github.com/PyTorchLightning/lightning-bolts/blob/2415b49a2b405693cd499e09162c89f807abbdc4/pl_bolts/datamodules/mnist_datamodule.py#L100-L108
        """
        if self._normalize:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transform_lib.Compose([transform_lib.ToTensor()])
        return mnist_transforms
