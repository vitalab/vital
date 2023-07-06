from argparse import ArgumentParser
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from vital.data.camus.config import CamusTags, Label, in_channels
from vital.data.camus.dataset import Camus
from vital.data.config import DataParameters, ProtoLabel, Subset
from vital.data.data_module import VitalDataModule
from vital.data.mixins import StructuredDataMixin


class CamusDataModule(StructuredDataMixin, VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the CAMUS dataset."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        labels: Sequence[ProtoLabel] = Label,
        fold: int = 5,
        use_sequence: bool = False,
        num_neighbors: int = 0,
        neighbor_padding: Literal["edge", "wrap"] = "edge",
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            labels: Labels of the segmentation classes to take into account (including background). If None, target all
                labels included in the data.
            fold: ID of the cross-validation fold to use.
            use_sequence: Enable use of full temporal sequences.
            num_neighbors: Number of neighboring frames on each side of an item's frame to include as part of an item's
                data.
            neighbor_padding: Mode used to determine how to pad neighboring instants at the beginning/end of a sequence.
                The options mirror those of the ``mode`` parameter of ``numpy.pad``.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        dataset_path = Path(dataset_path)
        labels = tuple(Label.from_proto_labels(labels))

        # Infer the shape of the data from the content of the dataset.
        try:
            # First try to get the first item from the training set
            image_shape = Camus(dataset_path, fold, Subset.TRAIN)[0][CamusTags.gt].shape
        except IndexError:
            # If there is no training set, try to get the first item from the testing set
            image_shape = Camus(dataset_path, fold, Subset.TEST)[0][CamusTags.gt].shape

        super().__init__(
            data_params=DataParameters(
                in_shape=(in_channels, *image_shape), out_shape=(len(labels), *image_shape), labels=labels
            ),
            **kwargs,
        )

        self._dataset_kwargs = {
            "path": dataset_path,
            "fold": fold,
            "labels": labels,
            "use_sequence": use_sequence,
            "neighbors": num_neighbors,
            "neighbor_padding": neighbor_padding,
        }

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        if stage == TrainerFn.FITTING:
            self.datasets[Subset.TRAIN] = Camus(image_set=Subset.TRAIN, **self._dataset_kwargs)
        if stage in [TrainerFn.FITTING, TrainerFn.VALIDATING]:
            self.datasets[Subset.VAL] = Camus(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == TrainerFn.TESTING:
            self.datasets[Subset.TEST] = Camus(image_set=Subset.TEST, **self._dataset_kwargs)
        if stage == TrainerFn.PREDICTING:
            self.datasets[Subset.PREDICT] = Camus(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)

    def group_ids(self, subset: Subset, level: Literal["patient", "view"] = "view") -> List[str]:
        """Lists the IDs of the different levels of groups/clusters samples in the data can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'view': all the data from the same view of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters samples in the data can belong to.
        """
        subset_dataset = self.datasets.get(subset, Camus(image_set=subset, **self._dataset_kwargs))
        return subset_dataset.list_groups(level=level)

    def predict_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.datasets[Subset.PREDICT], batch_size=None, num_workers=self.num_workers, pin_memory=True)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        """Override of generic ``add_argparse_args`` to manually add parser for arguments of custom types."""
        parser = super().add_argparse_args(parent_parser, **kwargs)

        # Hack to fetch the argument group created specifically for the data module's arguments
        dm_arg_group = [
            arg_group
            for arg_group in parser._action_groups
            if arg_group.title == f"{cls.__module__}.{cls.__qualname__}"
        ][0]

        # Add arguments w/ custom types not supported by Lightning's argparse creation tool
        dm_arg_group.add_argument(
            "--labels",
            type=Label.from_proto_label,
            default=tuple(Label),
            nargs="+",
            choices=tuple(Label),
            help="Labels of the segmentation classes to take into account (including background). "
            "If None, target all labels included in the data",
        )
        dm_arg_group.add_argument(
            "--neighbor_padding",
            type=str,
            choices=["edge", "wrap"],
            default="edge",
            help="Mode used to determine how to pad neighboring instants at the beginning/end of a sequence. The "
            "options mirror those of the ``mode`` parameter of ``numpy.pad``.",
        )

        return parent_parser
