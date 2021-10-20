from pathlib import Path
from typing import List, Literal, Sequence, Union, Tuple, Optional

from torch.utils.data import DataLoader

from vital.data.camus.config import CamusTags, Label, in_channels
from vital.data.camus.dataset import Camus
from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule
from vital.data.mixins import StructuredDataMixin


class CamusDataModule(StructuredDataMixin, VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the CAMUS dataset."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        labels: Sequence[Union[str, Label]] = Label,
        fold: int = 5,
        use_sequence: bool = False,
        max_patients: Optional[int] = None,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            labels: Labels of the segmentation classes to take into account (including background). If None, target all
                labels included in the data.
            fold: ID of the cross-validation fold to use.
            use_sequence: Enable use of full temporal sequences.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        dataset_path = Path(dataset_path)
        labels = tuple(Label.from_name(str(label)) for label in labels)
        self.max_patients = max_patients

        # Infer the shape of the data from the content of the dataset.
        try:
            # First try to get the first item from the training set
            image_shape = Camus(dataset_path, fold, Subset.TRAIN)[0][CamusTags.gt].shape
        except IndexError:
            # If there is no training set, try to get the first item from the testing set
            image_shape = Camus(dataset_path, fold, Subset.TEST)[0][CamusTags.gt].shape

        output_channels = 1 if len(labels) == 2 else len(labels)
        super().__init__(
            data_params=DataParameters(
                in_shape=(in_channels, *image_shape), out_shape=(output_channels, *image_shape), labels=labels
            ),
            **kwargs,
        )

        self._dataset_kwargs = {"path": dataset_path, "fold": fold, "labels": labels, "use_sequence": use_sequence}

    def setup(self, stage: Literal["fit", "test"]) -> None:  # noqa: D102
        if stage == "fit":
            self._dataset[Subset.TRAIN] = Camus(image_set=Subset.TRAIN, **self._dataset_kwargs,
                                                max_patients=self.max_patients)
            self._dataset[Subset.VAL] = Camus(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == "test":
            self._dataset[Subset.TEST] = Camus(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)

    def group_ids(self, subset: Subset, level: Literal["patient", "view"] = "view") -> List[str]:
        """Lists the IDs of the different levels of groups/clusters samples in the data can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'view': all the data from the same view of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters samples in the data can belong to.
        """
        subset_data = self.dataset().get(subset, Camus(image_set=subset, **self._dataset_kwargs))
        return subset_data.list_groups(level=level)

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
