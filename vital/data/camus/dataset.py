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
    """
    Args:
        use_sequence_index: whether to use instants' normalized indices in the sequence.
    """

    use_sequence_index: bool


class Camus(VisionDataset):
    def __init__(
        self,
        path: Path,
        image_set: Subset,
        labels: Sequence[Label],
        use_sequence: bool = False,
        use_sequence_index: bool = False,
        predict: bool = False,
        transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
        transform: Callable[[Tensor], Tensor] = None,
        target_transform: Callable[[Tensor], Tensor] = None,
    ):
        """
        Args:
            path: path to the HDF5 dataset.
            image_set: select the subset of images to use from the enumeration.
            labels: labels of the segmentation classes to take into account.
            use_sequence: whether to use the complete sequence between ED and ES for each view.
            use_sequence_index: whether to use instants' normalized indices in the sequence.
            predict: whether to receive the data in a format fit for inference (``True``) or training (``False``).
            transforms: a function that takes in an input/target pair and transforms them in a corresponding way
                        (only applied when `predict` is `False`, i.e. in train/validation mode).
            transform: a function that takes in an input and transforms it
                       (only applied when `predict` is `False`, i.e. in train/validation mode).
            target_transform: a function that takes in a target and transforms it
                              (only applied when `predict` is `False`, i.e. in train/validation mode).
        """
        super().__init__(path, transforms=transforms, transform=transform, target_transform=target_transform)
        self.image_set = image_set.value
        self.labels = labels
        self.use_sequence = use_sequence
        self.use_sequence_index = use_sequence_index

        with h5py.File(path, "r") as f:
            self.registered_dataset = f.attrs[CamusTags.registered]
            self.dataset_with_sequence = f.attrs[CamusTags.full_sequence]
        if self.use_sequence and not self.dataset_with_sequence:
            raise ValueError(
                "Request to use complete sequences, but the dataset only contains cardiac phase end instants. "
                "Should specify `no_sequence` flag, or generate a new dataset with sequences."
            )

        # Determine labels to remove based on labels to take into account
        self.labels_to_remove = [label for label in Label if label not in self.labels]

        # Determine whether to return data in a format suitable for training or inference
        if predict:
            self.item_list = self._get_patient_paths()
            self.getter = self._get_test_item
        else:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_train_item

    def __getitem__(self, index) -> Union[Dict[str, Tensor], Dict[View, Tuple[Tensor, Tensor]]]:
        return self.getter(index)

    def __len__(self):
        return len(self.item_list)

    def get_num_classes(self) -> int:
        return len(self.labels)

    def _get_patient_paths(self) -> List[str]:
        """Lists paths to the patients, from the requested ``image_set``, inside the HDF5 file.

        Returns:
            paths to the patients, from the requested ``image_set``, inside the HDF5 file.
        """
        with h5py.File(self.root, "r") as dataset:
            patient_paths = [f"{self.image_set}/{patient_id}" for patient_id in dataset[self.image_set].keys()]
        return patient_paths

    def _get_instant_paths(self) -> List[Tuple[str, int]]:
        """Lists paths to the instants, from the requested ``image_set``, inside the HDF5 file.

        Returns:
            paths to the instants, from the requested ``image_set``, inside the HDF5 file.
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
            index: index of the train/val sample in the train/val set's ``item_list``.

        Returns:
            data for training on a train/val item.
        """
        set_patient_view_key, instant = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            # Collect and process data
            view_imgs, view_gts = self._get_data(dataset, set_patient_view_key, CamusTags.img_proc, CamusTags.gt_proc)
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
            index: index of the test sample in the test set's ``item_list``.

        Returns:
            data for inference on a test item.
        """
        views = {}
        with h5py.File(self.root, "r") as dataset:
            for view in dataset[self.item_list[index]]:
                set_patient_view_key = f"{self.item_list[index]}/{view}"
                view = View(view)

                # Collect and process data
                proc_imgs, proc_gts = Camus._get_data(
                    dataset, set_patient_view_key, CamusTags.img_proc, CamusTags.gt_proc
                )
                proc_gts = self._process_target_data(proc_gts)

                # Indicate indices of instants with manually annotated segmentations in view sequences
                instants_with_gt = {
                    instant: Camus._get_metadata(dataset, set_patient_view_key, instant.value) for instant in Instant
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

        Args:
            index: index of the patient in the test set's ``item_list``.

        Returns:
            data about a patient.
        """
        with h5py.File(self.root, "r") as dataset:
            patient_data = PatientData(id=self.item_list[index].split("/")[1])
            for view in dataset[self.item_list[index]]:
                set_patient_view_key = f"{self.item_list[index]}/{view}"
                view = View(view)

                # Collect data
                gts = self._process_target_data(Camus._get_data(dataset, set_patient_view_key, CamusTags.gt))

                # Collect metadata
                info = Camus._get_metadata(dataset, set_patient_view_key, CamusTags.info)

                # Indicate indices of instants with manually annotated segmentations in view sequences
                instants_with_gt = {
                    instant: Camus._get_metadata(dataset, set_patient_view_key, instant.value) for instant in Instant
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
                        reg_step: Camus._get_metadata(dataset, set_patient_view_key, reg_step)
                        for reg_step in CamusRegisteringTransformer.registering_steps
                    }

                patient_data.views[view] = ViewData(gts, info, instants_with_gt, registering=registering_parameters)

        return patient_data

    @squeeze
    def _process_target_data(self, *args: np.ndarray) -> List[np.ndarray]:
        """Processes the target data to only keep requested labels and outputs them in categorical format.

        Args:
            *args: target data arrays to process.

        Returns:
            target data arrays processed and formatted.
        """
        return [
            remove_labels(
                target_data, [lbl.value for lbl in self.labels_to_remove], fill_label=Label.BG.value
            ).squeeze()
            for target_data in args
        ]

    @staticmethod
    @squeeze
    def _get_data(file: h5py.File, set_patient_view_key: str, *data_tags: str) -> List[np.ndarray]:
        """Fetches the requested data for a specific set/patient/view dataset from the HDF5 file.

        Args:
            file: the HDF5 dataset file.
            set_patient_view_key: the `set/patient/view` access path of the desired view group.
            *data_tags: names of the datasets to fetch from the view.

        Returns:
            dataset content for each tag passed in the parameters.
        """
        set_patient_view = file[set_patient_view_key]
        return [set_patient_view[data_tag][()] for data_tag in data_tags]

    @staticmethod
    @squeeze
    def _get_metadata(file: h5py.File, set_patient_view_key: str, *metadata_tags: str) -> List[np.ndarray]:
        """Fetches the requested metadata for a specific set/patient/view dataset from the HDF5 file.

        Args:
            file: the HDF5 dataset file.
            set_patient_view_key: the `set/patient/view` access path of the desired view group.
            *metadata_tags: names of attributes to fetch from the view.

        Returns:
            attribute values for each tag passed in the parameters.
        """
        set_patient_view = file[set_patient_view_key]
        return [set_patient_view.attrs[attr_tag] for attr_tag in metadata_tags]