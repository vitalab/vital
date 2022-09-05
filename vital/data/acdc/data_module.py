from pathlib import Path
from typing import Optional, Union

from pytorch_lightning.trainer.states import TrainerFn

from vital.data.acdc.config import Label, image_size, in_channels
from vital.data.acdc.dataset import Acdc
from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule


class AcdcDataModule(VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the ACDC dataset."""

    def __init__(self, dataset_path: Union[str, Path], use_da: bool = True, **kwargs):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            use_da: Enable use of data augmentation.
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

        self._dataset_kwargs = {"path": Path(dataset_path), "use_da": use_da}

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        if stage == TrainerFn.FITTING:
            self.datasets[Subset.TRAIN] = Acdc(image_set=Subset.TRAIN, **self._dataset_kwargs)
        if stage in [TrainerFn.FITTING, TrainerFn.VALIDATING]:
            self.datasets[Subset.VAL] = Acdc(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == TrainerFn.TESTING:
            self.datasets[Subset.TEST] = Acdc(image_set=Subset.TEST, **self._dataset_kwargs)
