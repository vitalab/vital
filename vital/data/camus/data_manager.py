from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Literal

from torch.utils.data import DataLoader

from vital.data.camus.config import Label, image_size, in_channels
from vital.data.camus.dataset import Camus, DataParameters
from vital.data.config import Subset
from vital.data.mixins import StructuredDataMixin
from vital.systems.vital_system import SystemDataManagerMixin


class CamusSystemDataManagerMixin(StructuredDataMixin, SystemDataManagerMixin):
    """Implementation of the mixin handling the training/validation/testing phases for the CAMUS dataset."""

    use_da: bool = False  #: Whether the system applies Data Augmentation (DA) by default
    use_sequence: bool = False  #: Whether the system uses complete sequences by default

    def __init__(self, **kwargs):
        """Handles initializing parameters related to the nature of the data.

        Args:
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Propagate data_params to allow model to adapt to data config
        # Overrides saved data_params for models loaded from a checkpoint
        kwargs["data_params"] = DataParameters(
            in_shape=(in_channels, image_size, image_size),
            out_shape=(len(kwargs["labels"]), image_size, image_size),
        )
        super().__init__(**kwargs)

        self.labels = [str(label) for label in self.hparams.labels]
        self.dataset: Dict[Subset, Camus] = {}
        self._dataset_kwargs = {
            "path": self.hparams.dataset_path,
            "fold": self.hparams.fold,
            "labels": self.hparams.labels,
            "use_sequence": self.hparams.use_sequence,
        }

    def setup(self, stage: Literal["fit", "test"]) -> None:  # noqa: D102
        if stage == "fit":
            self.dataset[Subset.TRAIN] = Camus(image_set=Subset.TRAIN, **self._dataset_kwargs)
            self.dataset[Subset.VAL] = Camus(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == "test":
            self.dataset[Subset.TEST] = Camus(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)

    def train_group_ids(self, level: Literal["patient", "view"] = "view") -> List[str]:
        """Lists the IDs of the different levels of groups/clusters samples in the training data can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'view': all the data from the same view of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters samples in the training data can belong to.
        """
        train_data = self.dataset.get(Subset.TRAIN, Camus(image_set=Subset.TRAIN, **self._dataset_kwargs))
        return train_data.list_groups(level=level)

    def val_group_ids(self, level: Literal["patient", "view"] = "view") -> List[str]:
        """Lists the IDs of the different levels of groups/clusters samples in the validation data can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'view': all the data from the same view of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters samples in the validation data can belong to.
        """
        val_data = self.dataset.get(Subset.VAL, Camus(image_set=Subset.VAL, **self._dataset_kwargs))
        return val_data.list_groups(level=level)

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
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.on_gpu,
        )

    @classmethod
    def add_data_manager_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        parser = super().add_data_manager_args(parser)
        parser.add_argument("dataset_path", type=Path, help="Path to the HDF5 dataset")
        parser.add_argument("--fold", type=int, default=5, help="ID of the cross-validation fold to use")
        parser.add_argument(
            "--labels",
            type=Label.from_name,
            default=list(Label),
            nargs="+",
            choices=list(Label),
            help="Labels of the segmentation classes to take into account (including background). "
            "If None, target all labels included in the data",
        )

        if cls.use_sequence:
            parser.add_argument(
                "--no_sequence", dest="use_sequence", action="store_false", help="Disable use of interpolated sequences"
            )
        else:
            parser.add_argument(
                "--use_sequence", dest="use_sequence", action="store_true", help="Enable use of interpolated sequences"
            )

        return parser
