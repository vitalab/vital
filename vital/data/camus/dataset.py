import itertools
from pathlib import Path
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import to_tensor

from vital.data.camus.config import CamusTags, Label
from vital.data.camus.data_struct import ViewMetadata
from vital.data.camus.utils.register import CamusRegisteringTransformer
from vital.data.config import Subset
from vital.utils.decorators import squeeze
from vital.utils.image.transform import remove_labels, segmentation_to_tensor
from vital.utils.image.us.measure import EchoMeasure

ItemId = Tuple[str, int]
InstantItem = Dict[str, Union[str, Tensor]]
RecursiveInstantItem = Dict[str, Union[str, Tensor, Dict[str, InstantItem]]]


class Camus(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the CAMUS dataset."""

    def __init__(
        self,
        path: Path,
        fold: int,
        image_set: Subset,
        labels: Sequence[Label] = Label,
        use_sequence: bool = False,
        predict: bool = False,
        neighbors: Union[int, Sequence[int]] = 0,
        neighbor_padding: Literal["edge", "wrap"] = "edge",
        transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
        transform: Callable[[Tensor], Tensor] = None,
        target_transform: Callable[[Tensor], Tensor] = None,
    ):
        """Initializes class instance.

        Args:
            path: Path to the HDF5 dataset.
            fold: ID of the cross-validation fold to use.
            image_set: Subset of images to use.
            labels: Labels of the segmentation classes to take into account.
            use_sequence: Whether to use the complete sequence between ED and ES for each view.
            predict: Whether to receive the data in a format fit for inference (``True``) or training (``False``).
            neighbors: Neighboring frames to include in a train/val item. The value either indicates the number of
                neighboring frames on each side of the item's frame (`int`), or a list of offsets w.r.t the item's
                frame (`Sequence[int]`).
            neighbor_padding: Mode used to determine how to pad neighboring instants at the beginning/end of a sequence.
                The options mirror those of the ``mode`` parameter of ``numpy.pad``.
            transforms: Function that takes in an input/target pair and transforms them in a corresponding way.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
            transform: Function that takes in an input and transforms it.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
            target_transform: Function that takes in a target and transforms it.
                (only applied when `predict` is `False`, i.e. in train/validation mode)

        Raises:
            RuntimeError: If flags/arguments are requested that cannot be provided by the HDF5 dataset.
                - ``use_sequence`` flag is active, while the HDF5 dataset doesn't include full sequences.
        """
        super().__init__(path, transforms=transforms, transform=transform, target_transform=target_transform)
        self.fold = fold
        self.image_set = image_set
        self.labels = labels
        self.use_sequence = use_sequence
        self.predict = predict
        self.neighbors = neighbors
        self.neighbor_padding = neighbor_padding

        with h5py.File(path, "r") as f:
            self.registered_dataset = f.attrs[CamusTags.registered]
            self.dataset_with_sequence = f.attrs[CamusTags.full_sequence]
        if self.use_sequence and not self.dataset_with_sequence:
            raise RuntimeError(
                "Request to use complete sequences, but the dataset only contains cardiac phase end instants. "
                "Should specify `no_sequence` flag, or generate a new dataset with sequences."
            )

        # Determine labels to remove based on labels to take into account
        self.labels_to_remove = [label for label in Label if label not in self.labels]

        # Determine whether to return data in a format suitable for training or inference
        if self.predict:
            self.item_list = self.list_groups(level="view")
            self.getter = self._get_predict_item
        else:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_train_item

    def __getitem__(self, index) -> Union[RecursiveInstantItem, Dict[str, Tensor]]:
        """Fetches an item, whose structure depends on the ``predict`` value, from the internal list of items.

        Notes:
            - When in ``predict`` mode (i.e. for test-time inference), an item corresponds to the views' ultrasound
              images and groundtruth segmentations for a patient.
            - When not in ``predict`` mode (i.e. during training), an item corresponds to an image/segmentation pair for
              a single frame.

        Args:
            index: Index of the item to fetch from the internal sequence of items.

        Returns:
            Item from the internal list at position ``index``.
        """
        return self.getter(index)

    def __len__(self):  # noqa: D105
        return len(self.item_list)

    def list_groups(self, level: Literal["patient", "view"] = "view") -> List[str]:
        """Lists the paths of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'view': all the data from the same view of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.
        """
        with h5py.File(self.root, "r") as dataset:
            # List the patients
            groups = [
                patient_path_byte.decode()
                for patient_path_byte in dataset[f"cross_validation/fold_{self.fold}/{self.image_set}"]
            ]
            if level == "view":
                groups = [f"{patient}/{view}" for patient in groups for view in dataset[patient].keys()]

        return groups

    def _get_instant_paths(self) -> List[ItemId]:
        """Lists paths to the instants, from the requested ``self.image_set``, inside the HDF5 file.

        Returns:
            Paths to the instants, from the requested ``self.image_set``, inside the HDF5 file.
        """

        def include_image(view_group: h5py.Group, instant: int) -> bool:
            is_clinically_important_instant = instant in (
                view_group.attrs[instant_key] for instant_key in view_group.attrs[CamusTags.instants]
            )
            return (
                not self.dataset_with_sequence
                or self.use_sequence
                or (self.dataset_with_sequence and is_clinically_important_instant)
            )

        image_paths = []
        view_paths = self.list_groups(level="view")
        with h5py.File(self.root, "r") as dataset:
            for view_path in view_paths:
                view_group = dataset[view_path]
                for instant in range(view_group[CamusTags.gt].shape[0]):
                    if include_image(view_group, instant):
                        image_paths.append((view_path, instant))
        return image_paths

    def _get_train_item(self, index: int) -> RecursiveInstantItem:
        """Fetches data required for training on a train/val item.

        Notes:
            -  If ``self.neighbors>0``, the result will include data from other items: neighboring instants from the
               patient/view sequence of the item pointed at by ``index``. Because of this, setting ``self.neighbors>0``
               can rapidly increase the memory footprint of the program.

        Args:
            index: Index of the train/val sample in the train/val set's ``self.item_list``.

        Returns:
            Data for training on a train/val item.
        """
        item = self._get_instant_item(index)
        if self.neighbors:
            patient_view_key, instant = self.item_list[index]
            # Determine the requested neighbors' offset to the current item
            if isinstance(self.neighbors, int):  # If `neighbors` indicates the number of neighbors on each side
                instant_diffs = itertools.chain(range(-self.neighbors, 0), range(1, self.neighbors + 1))
            else:  # If `neighbors` is a list of the neighbors offset w.r.t the current item
                instant_diffs = self.neighbors

            # Determine which items' to use as neighbors
            with h5py.File(self.root, "r") as dataset:
                seq_len = len(dataset[patient_view_key][CamusTags.img_proc])
            if self.neighbor_padding == "edge":
                neigh_instants = {diff: np.clip(instant + diff, 0, seq_len) for diff in instant_diffs}
            elif self.neighbor_padding == "wrap":
                neigh_instants = {diff: (instant + diff) % seq_len for diff in instant_diffs}
            else:
                raise ValueError(
                    f"Unexpected value for `neighbor_padding`: {self.neighbor_padding}. Use one of: ('edge', 'wrap')"
                )

            # Fetch the neighbors' data
            item[CamusTags.neighbors] = {
                diff: self._get_instant_item(self.item_list.index((patient_view_key, instant)))
                for diff, instant in neigh_instants.items()
            }

        return item

    def _get_instant_item(self, index: int) -> InstantItem:
        """Fetches data and metadata related to an instant (single image/groundtruth pair + metadata).

        Args:
            index: Index of the instant sample in the dataset's ``self.item_list``.

        Returns:
            Data and metadata related to an instant.
        """
        patient_view_key, instant = self.item_list[index]

        # Collect data
        with h5py.File(self.root, "r") as dataset:
            view_imgs, view_gts = self._get_data(dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc)

        # Format data
        img = view_imgs[instant]
        gt = self._process_target_data(view_gts[instant])
        img, gt = to_tensor(img), segmentation_to_tensor(gt)

        # Apply transforms on the data
        if self.transforms:
            img, gt = self.transforms(img, gt)

        # Compute attributes on the data
        frame_pos = torch.tensor([instant / len(view_imgs)])
        gt_attrs = get_segmentation_attributes(gt, self.labels)

        return {
            CamusTags.id: f"{patient_view_key}/{instant}",
            CamusTags.group: patient_view_key,
            CamusTags.img: img,
            CamusTags.gt: gt,
            CamusTags.frame_pos: frame_pos,
            **gt_attrs,
        }

    def _get_predict_item(self, index: int) -> Dict[str, Tensor]:
        """Fetches data required for inference on a prediction item, i.e. a view.

        Args:
            index: Index of the prediction sample in the dataset's ``self.item_list``.

        Returns:
            Data related a to a prediction item, i.e. a view.
        """
        patient_view_key = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            # Collect data
            view_imgs, view_gts = self._get_data(dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc)
            view_gts = self._process_target_data(view_gts)

            # Collect metadata
            voxelspacing, clinically_important_instants = Camus._get_metadata(
                dataset, patient_view_key, CamusTags.voxelspacing, CamusTags.instants
            )
            instants = {
                instant: Camus._get_metadata(dataset, patient_view_key, instant)
                for instant in clinically_important_instants
            }
            full_resolution_gts = Camus._get_data(dataset, patient_view_key, CamusTags.gt)
            full_resolution_gts = self._process_target_data(full_resolution_gts)

            # If we do not use the whole sequence
            if self.dataset_with_sequence and not self.use_sequence:
                # Only keep clinically important instants
                instant_indices = list(instants.values())
                view_imgs = view_imgs[instant_indices]
                view_gts = view_gts[instant_indices]
                full_resolution_gts = full_resolution_gts[instant_indices]

                # Update indices of clinically important instants to match the new slicing of the sequences
                instants = {instant: idx for idx, instant in enumerate(instants)}

            # Extract metadata concerning the registering applied
            registering_parameters = None
            if self.registered_dataset:
                registering_parameters = {
                    reg_step: Camus._get_metadata(dataset, patient_view_key, reg_step)
                    for reg_step in CamusRegisteringTransformer.registering_steps
                }

        # Format data
        view_imgs = torch.stack([to_tensor(view_img) for view_img in view_imgs])
        view_gts = torch.stack([segmentation_to_tensor(view_gt) for view_gt in view_gts])

        # Apply transforms on the data
        if self.transforms:
            view_imgs, view_gts = self.transforms(view_imgs, view_gts)

        # Compute attributes on the data
        frame_pos = torch.linspace(0, 1, len(view_imgs))
        gts_attrs = get_segmentation_attributes(view_gts, self.labels)

        # Build the batch to return
        view_metadata = ViewMetadata(
            gt=full_resolution_gts, voxelspacing=voxelspacing, instants=instants, registering=registering_parameters
        )
        return {
            CamusTags.id: patient_view_key,
            CamusTags.group: patient_view_key,
            CamusTags.img: view_imgs,
            CamusTags.gt: view_gts,
            CamusTags.frame_pos: frame_pos,
            **gts_attrs,
            CamusTags.metadata: view_metadata,
        }

    @squeeze
    def _process_target_data(self, *args: np.ndarray) -> List[np.ndarray]:
        """Processes the target data to only keep requested labels and outputs them in categorical format.

        Args:
            *args: Target data arrays to process.

        Returns:
            Target data arrays processed and formatted.
        """
        return [
            remove_labels(target_data, self.labels_to_remove, fill_label=Label.BG).squeeze() for target_data in args
        ]

    @staticmethod
    @squeeze
    def _get_data(file: h5py.File, patient_view_key: str, *data_tags: str) -> List[np.ndarray]:
        """Fetches the requested data for a specific patient/view dataset from the HDF5 file.

        Args:
            file: HDF5 dataset file.
            patient_view_key: `patient/view` access path of the desired view group.
            *data_tags: Names of the datasets to fetch from the view.

        Returns:
            Dataset content for each tag passed in the parameters.
        """
        patient_view = file[patient_view_key]
        return [patient_view[data_tag][()] for data_tag in data_tags]

    @staticmethod
    @squeeze
    def _get_metadata(file: h5py.File, patient_view_key: str, *metadata_tags: str) -> List[np.ndarray]:
        """Fetches the requested metadata for a specific patient/view dataset from the HDF5 file.

        Args:
            file: HDF5 dataset file.
            patient_view_key: `patient/view` access path of the desired view group.
            *metadata_tags: Names of attributes to fetch from the view.

        Returns:
            Attribute values for each tag passed in the parameters.
        """
        patient_view = file[patient_view_key]
        return [patient_view.attrs[attr_tag] for attr_tag in metadata_tags]


def get_segmentation_attributes(
    segmentation: Union[np.ndarray, Tensor], labels: Sequence[Label]
) -> Dict[str, Union[np.ndarray, Tensor]]:
    """Measures a variety of attributes on a (batch of) segmentation(s).

    Args:
        segmentation: ([N], H, W), Segmentation(s) on which to compute a variety of attributes.
        labels: Labels of the classes included in the segmentation(s).

    Returns:
        Mapping between the attributes and arrays of shape ([N]) of their values for each segmentation in the batch.
    """
    attrs = {}
    if Label.LV in labels:
        attrs.update(
            {
                CamusTags.lv_area: EchoMeasure.structure_area(segmentation, labels=Label.LV),
                CamusTags.lv_orientation: EchoMeasure.structure_orientation(
                    segmentation, labels=Label.LV, reference_orientation=90
                ),
            }
        )
    if Label.MYO in labels:
        attrs.update({CamusTags.myo_area: EchoMeasure.structure_area(segmentation, labels=Label.MYO)})
    if Label.LV in labels and Label.MYO in labels:
        attrs.update(
            {
                CamusTags.lv_base_width: EchoMeasure.lv_base_width(segmentation, Label.LV, Label.MYO),
                CamusTags.lv_length: EchoMeasure.lv_length(segmentation, Label.LV, Label.MYO),
                CamusTags.epi_center_x: EchoMeasure.structure_center(
                    segmentation, labels=[Label.LV, Label.MYO], axis=1
                ),
                CamusTags.epi_center_y: EchoMeasure.structure_center(
                    segmentation, labels=[Label.LV, Label.MYO], axis=0
                ),
            }
        )
    if Label.ATRIUM in labels:
        attrs.update({CamusTags.atrium_area: EchoMeasure.structure_area(segmentation, labels=Label.ATRIUM)})
    return attrs
