from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Literal

from torch.utils.data import DataLoader

from vital.data.config import DataParameters, Subset
from vital.data.mri.acdc.dataset import Acdc
from vital.data.mri.config import Label, image_size, in_channels
from vital.systems.system import SystemDataManagerMixin


class AcdcSystemDataManagerMixin(SystemDataManagerMixin):
    """Implementation of the mixin handling the training/validation/testing phases for the ACDC dataset."""

    use_da: bool = True  #: Whether the system applies Data Augmentation (DA) by default

    def __init__(self, **kwargs):
        """Handles initializing parameters related to the nature of the data.

        Args:
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Propagate data_params to allow model to adapt to data config
        # Overrides saved data_params for models loaded from a checkpoint
        kwargs["data_params"] = DataParameters(
            in_shape=(in_channels, image_size, image_size),
            out_shape=(len(list(Label)), image_size, image_size),
        )
        super().__init__(**kwargs)

        self.dataset: Dict[Subset, Acdc] = {}
        self.labels = [str(label) for label in list(Label)]
        self._dataset_kwargs = {
            "path": self.hparams.dataset_path,
            "use_da": self.hparams.use_da,
        }

    def setup(self, stage: Literal["fit", "test"]) -> None:  # noqa: D102
        if stage == "fit":
            self.dataset[Subset.TRAIN] = Acdc(image_set=Subset.TRAIN, **self._dataset_kwargs)
            self.dataset[Subset.VAL] = Acdc(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == "test":
            self.dataset[Subset.TEST] = Acdc(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.dataset[Subset.TRAIN],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.dataset[Subset.VAL],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.dataset[Subset.TEST],
            batch_size=None,  # batch_size=None returns one full patient at each step.
            num_workers=self.hparams.num_workers,
            pin_memory=self.on_gpu,
        )

    @classmethod
    def add_data_manager_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        parser = super().add_data_manager_args(parser)
        parser.add_argument("dataset_path", type=Path, help="Path to the HDF5 dataset")
        if cls.use_da:
            parser.add_argument("--no_da", dest="use_da", action="store_false", help="Disable use of data augmentation")
        else:
            parser.add_argument("--use_da", dest="use_da", action="store_true", help="Enable use of data augmentation")
        return parser
