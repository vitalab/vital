from pathlib import Path
from typing import Callable, Tuple
from typing import Literal, Union, Optional

from torch import Tensor
from torch.utils.data import DataLoader
from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule
from vital.data.lung_xray.config import Label
from vital.data.lung_xray.dataset import LungXRay
from vital.data.mixins import StructuredDataMixin


class LungXRayDataModule(StructuredDataMixin, VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the Lung X-ray dataset."""

    def __init__(
            self,
            dataset_path: Union[str, Path],
            transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
            transform: Callable[[Tensor], Tensor] = None,
            target_transform: Callable[[Tensor], Tensor] = None,
            max_patients: Optional[int] = None,
            da: Literal["pixel", "spatial"] = None,
            **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        dataset_path = Path(dataset_path)
        self.max_patients = max_patients
        self.data_augmentation = da

        image_shape = (256, 256)
        in_channels = 1

        super().__init__(
            data_params=DataParameters(
                in_shape=(in_channels, *image_shape), out_shape=(1, *image_shape), labels=tuple(Label)
            ),
            **kwargs,
        )

        self._dataset_kwargs = {
            "path": dataset_path,
            'transforms': transforms,
            'transform': transform,
            'target_transform': target_transform
        }

    def setup(self, stage: Literal["fit", "test"]) -> None:  # noqa: D102
        if stage == "fit":
            self._dataset[Subset.TRAIN] = LungXRay(image_set=Subset.TRAIN, **self._dataset_kwargs,
                                                   max_patients=self.max_patients,
                                                   data_augmentation=self.data_augmentation)
            self._dataset[Subset.VAL] = LungXRay(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == "test":
            self._dataset[Subset.TEST] = LungXRay(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)

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
            self.dataset(subset=Subset.TEST), batch_size=None, num_workers=self.num_workers, pin_memory=True
        )
