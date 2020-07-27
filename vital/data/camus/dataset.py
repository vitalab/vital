from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import to_tensor

import vital
from vital.data.camus.config import CamusTags, Instant, Label, View
from vital.data.camus.data_struct import PatientData, ViewData
from vital.data.config import Subset
from vital.utils.decorators import squeeze
from vital.utils.image.register.camus import CamusRegisteringTransformer
from vital.utils.image.transform import remove_labels, segmentation_to_tensor
from vital.utils.parameters import parameters


@parameters
class DataParameters(vital.utils.parameters.DataParameters):
    """Extension of the generic ``DataParameters`` dataclass for CAMUS-specific parameters.

    Args:
        in_shape: (height, width, channels) shape of the input data.
        out_shape: (height, width, channels) shape of the target data.
        use_sequence_index: Whether to use instants' normalized indices in the sequence.
    """

    use_sequence_index: bool


class Camus(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the CAMUS dataset."""

    def __init__(
        self,
        path: Path,
        fold: int,
        image_set: Subset,
        labels: Sequence[Label],
        use_sequence: bool = False,
        use_sequence_index: bool = False,
        predict: bool = False,
        transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
        transform: Callable[[Tensor], Tensor] = None,
        target_transform: Callable[[Tensor], Tensor] = None,
    ):  # noqa: D205,D212,D415
        """
        Args:
            path: Path to the HDF5 dataset.
            fold: ID of the cross-validation fold to use.
            image_set: Subset of images to use.
            labels: Labels of the segmentation classes to take into account.
            use_sequence: Whether to use the complete sequence between ED and ES for each view.
            use_sequence_index: Whether to use instants' normalized indices in the sequence.
            predict: Whether to receive the data in a format fit for inference (``True``) or training (``False``).
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
        self.image_set = image_set.value
        self.labels = labels
        self.use_sequence = use_sequence
        self.use_sequence_index = use_sequence_index
        self.predict = predict

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
            self.item_list = self._get_patient_paths()
            self.getter = self._get_test_item
        else:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_train_item

    def __getitem__(self, index) -> Union[Dict[str, Tensor], Dict[View, Tuple[Tensor, Tensor]]]:
        """Fetches an item, whose structure depends on the ``predict`` value, from the internal list of items.

        Notes:
            - When in ``predict`` mode (i.e. for test-time inference), an item corresponds to the views' ultrasound
              images and groundtruth segmentations for a patient.
            - When not in ``predict`` mode (i.e. during training), an item corresponds to a image/segmentation pair for
              a single frame.

        Args:
            index: Index of the item to fetch from the internal sequence of items.

        Returns:
            Item from the internal list at position ``index``.
        """
        return self.getter(index)

    def __len__(self):  # noqa: D105
        return len(self.item_list)

    def get_num_classes(self) -> int:
        """Counts the number of segmentation classes present in the dataset."""
        return len(self.labels)

    def _get_patient_paths(self) -> List[str]:
        """Lists paths to the patients, from the requested ``self.image_set``, inside the HDF5 file.

        Returns:
            Paths to the patients, from the requested ``self.image_set``, inside the HDF5 file.
        """
        with h5py.File(self.root, "r") as dataset:
            patient_paths = [
                patient_path_byte.decode()
                for patient_path_byte in dataset[f"cross_validation/fold_{self.fold}/{self.image_set}"]
            ]
        return patient_paths

    def _get_instant_paths(self) -> List[Tuple[str, int]]:
        """Lists paths to the instants, from the requested ``self.image_set``, inside the HDF5 file.

        Returns:
            Paths to the instants, from the requested ``self.image_set``, inside the HDF5 file.
        """

        def include_image(view_group: h5py.Group, instant: int) -> bool:
            is_instant_with_gt = instant in (view_group.attrs[instant_key] for instant_key in Instant.values())
            return (
                not self.dataset_with_sequence
                or self.use_sequence
                or (self.dataset_with_sequence and is_instant_with_gt)
            )

        image_paths = []
        patient_paths = self._get_patient_paths()
        with h5py.File(self.root, "r") as dataset:
            for patient_path in patient_paths:
                for view in dataset[patient_path].keys():
                    view_group = dataset[patient_path][view]
                    for instant in range(view_group[CamusTags.gt].shape[0]):
                        if include_image(view_group, instant):
                            image_paths.append((f"{patient_path}/{view}", instant))
        return image_paths

    def _get_train_item(self, index: int) -> Dict[str, Tensor]:
        """Fetches data required for training on a train/val item (single image/groundtruth pair).

        Args:
            index: Index of the train/val sample in the train/val set's ``self.item_list``.

        Returns:
            Data for training on a train/val item.
        """
        patient_view_key, instant = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            # Collect and process data
            view_imgs, view_gts = self._get_data(dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc)
            img = view_imgs[instant]
            gt = self._process_target_data(view_gts[instant])

            # Collect metadata
            # Explicit cast to float32 to avoid "Expected object" type error in PyTorch models
            # that output ``FloatTensor`` by default (and not ``DoubleTensor``)
            sequence_idx = np.float32(instant / view_imgs.shape[0])

        img, gt = to_tensor(img), segmentation_to_tensor(gt)
        if self.transforms:
            img, gt = self.transforms(img, gt)

        item = {CamusTags.img: img, CamusTags.gt: gt}

        if self.use_sequence_index:
            item[CamusTags.regression] = sequence_idx

        return item

    def _get_test_item(self, index: int) -> Dict[View, Tuple[Tensor, Tensor]]:
        """Fetches data required for inference on a test item (whole patient).

        Args:
            index: Index of the test sample in the test set's ``self.item_list``.

        Returns:
            Data for inference on a test item.
        """
        views = {}
        with h5py.File(self.root, "r") as dataset:
            for view in dataset[self.item_list[index]]:
                patient_view_key = f"{self.item_list[index]}/{view}"
                view = View(view)

                # Collect and process data
                proc_imgs, proc_gts = Camus._get_data(dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc)
                proc_gts = self._process_target_data(proc_gts)

                # Indicate indices of instants with manually annotated segmentations in view sequences
                instants_with_gt = {
                    instant: Camus._get_metadata(dataset, patient_view_key, instant.value) for instant in Instant
                }

                # Only keep instants with manually annotated groundtruths if we do not use the whole sequence
                if self.dataset_with_sequence and not self.use_sequence:
                    proc_imgs = proc_imgs[list(instants_with_gt.values())]
                    proc_gts = proc_gts[list(instants_with_gt.values())]

                # Transform arrays to tensor
                proc_imgs_tensor = torch.stack([to_tensor(proc_img) for proc_img in proc_imgs])
                proc_gts_tensor = torch.stack([segmentation_to_tensor(proc_gt) for proc_gt in proc_gts])

                views[view] = (proc_imgs_tensor, proc_gts_tensor)

        return views

    def get_patient_data(self, index: int) -> PatientData:
        """Fetches data about a patient that is typically not used by models for inference.

        Returns additional data about the same patient as ``get_test_item`` returns inference-necessary data.
        The additional data returned by ``get_patient_data`` should be useful during evaluation
        (e.g. to save along with the predictions).

        Notes:
            - This method should only be used on datasets in ``predict`` mode. This is because items correspond to
              patients in those datasets, ``get_patient_data`` works in pair with directly indexing the dataset to fetch
              data about the ith patient (i.e. the index means the same for both methods). For datasets not in
              ``predict`` mode, where items don't correspond to patients, the `index` parameter makes no sense.

        Args:
            index: Index of the patient in the dataset's ``self.item_list``.

        Returns:
            Data about a patient.

        Raises:
            RuntimeError: If ``get_patient_data`` is called on a ``Camus`` instance not in ``predict`` mode.
        """
        if not self.predict:
            raise RuntimeError("Method `get_patient_data` should only be used on datasets in `predict` mode.")

        with h5py.File(self.root, "r") as dataset:
            patient_data = PatientData(id=self.item_list[index])
            for view in dataset[self.item_list[index]]:
                patient_view_key = f"{self.item_list[index]}/{view}"
                view = View(view)

                # Collect data
                gts = self._process_target_data(Camus._get_data(dataset, patient_view_key, CamusTags.gt))

                # Collect metadata
                info = Camus._get_metadata(dataset, patient_view_key, CamusTags.info)

                # Indicate indices of instants with manually annotated segmentations in view sequences
                instants_with_gt = {
                    instant: Camus._get_metadata(dataset, patient_view_key, instant.value) for instant in Instant
                }

                # Only keep instants with manually annotated groundtruths if we do not use the whole sequence
                if self.dataset_with_sequence and not self.use_sequence:
                    gts = gts[list(instants_with_gt.values())]

                    # Update indices of instants with manually annotated segmentations in view sequences in
                    # newly sliced sequences
                    instants_with_gt = {instant: idx for idx, instant in enumerate(Instant)}

                # Extract metadata concerning the registering applied
                registering_parameters = None
                if self.registered_dataset:
                    registering_parameters = {
                        reg_step: Camus._get_metadata(dataset, patient_view_key, reg_step)
                        for reg_step in CamusRegisteringTransformer.registering_steps
                    }

                patient_data.views[view] = ViewData(gts, info, instants_with_gt, registering=registering_parameters)

        return patient_data

    @squeeze
    def _process_target_data(self, *args: np.ndarray) -> List[np.ndarray]:
        """Processes the target data to only keep requested labels and outputs them in categorical format.

        Args:
            *args: Target data arrays to process.

        Returns:
            Target data arrays processed and formatted.
        """
        return [
            remove_labels(
                target_data, [lbl.value for lbl in self.labels_to_remove], fill_label=Label.BG.value
            ).squeeze()
            for target_data in args
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
