from pathlib import Path
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union

import albumentations as A
import cv2
import h5py
import numpy as np
import torch
import torchvision
from torch import Tensor

from vital.data.camus.config import CamusTags, Label
from vital.data.camus.data_struct import PatientData, ViewData
from vital.data.camus.utils.register import CamusRegisteringTransformer
from vital.data.config import Subset, TransformedSegmentationData
from vital.utils.decorators import squeeze
from vital.utils.image.transform import remove_labels, segmentation_to_tensor

ItemId = Tuple[str, int]


class Camus(torchvision.datasets.VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the CAMUS dataset."""

    def __init__(
        self,
        path: Path,
        fold: int,
        image_set: Subset,
        labels: Sequence[Label] = Label,
        use_sequence: bool = False,
        predict: bool = False,
        data_augmentation: Literal["pixel", "spatial"] = None,
        transforms: Sequence[Callable[[np.ndarray, np.ndarray], TransformedSegmentationData]] = None,
    ):
        """Initializes class instance.

        Args:
            path: Path to the HDF5 dataset.
            fold: ID of the cross-validation fold to use.
            image_set: Subset of images to use.
            labels: Labels of the segmentation classes to take into account.
            use_sequence: Whether to use the complete sequence between ED and ES for each view.
            predict: Whether to receive the data in a format fit for inference (``True``) or training (``False``).
            data_augmentation: Flag for the type of data augmentation to use (if any).
                - 'pixel': Pixel-level transforms of the input image (e.g. noise, normalization, blur, etc.) that leave
                   the target mask unchanged. Recommended for the standard segmentation task.
                - 'spatial': Spatial-level transforms that affect the input image and target mask simultaneously
                  (e.g. shift, scale, rotate). Not recommended for segmentation, but for representation learning task.
            transforms: Collection of function that take an image/mask pair and transform it in a corresponding way.
                (only applied when `use_da` is `True` and `predict` is `False`, i.e. in train/validation mode)

        Raises:
            RuntimeError: If flags/arguments are requested that cannot be provided by the HDF5 dataset.
                - ``use_sequence`` flag is active, while the HDF5 dataset doesn't include full sequences.
        """
        super().__init__(path)
        self.fold = fold
        self.image_set = image_set
        self.labels = labels
        self.use_sequence = use_sequence
        self.predict = predict

        if transforms is not None:
            self.transforms = A.Compose(transforms)
        elif data_augmentation:
            if data_augmentation == "pixel":
                # TODO: Fix normalization of intensity values in input image
                self.transforms = A.Compose(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                        A.RandomGamma(gamma_limit=(80, 120)),
                    ]
                )
            elif data_augmentation == "spatial":
                self.transforms = A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0
                )
            else:
                raise ValueError(
                    f"Unexpected value for parameter `data_augmentation`: {data_augmentation}. "
                    f"Please change it to one of the supported values: ['pixel', 'spatial']."
                )
        else:
            self.transforms = None

        self._base_transform = torchvision.transforms.ToTensor()
        self._base_target_transform = segmentation_to_tensor

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
            self.item_list = self.list_groups(level="patient")
            self.getter = self._get_test_item
        else:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_train_item

    def __getitem__(self, index) -> Union[Dict[str, Union[str, Tensor]], PatientData]:
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
                for patient_path_byte in dataset[f"cross_validation/fold_{self.fold}/{self.image_set.value}"]
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

    def _get_train_item(self, index: int) -> Dict[str, Union[str, Tensor]]:
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
            frame_pos = np.float32(instant / view_imgs.shape[0])

        # Data augmentation transforms applied before converting numpy array to torch tensor
        if self.transforms:
            transformed = self.transforms(image=img, mask=gt)
            img = transformed["image"]
            gt = transformed["mask"]

        img, gt = self._base_transform(img), self._base_target_transform(gt)
        frame_pos = torch.tensor([frame_pos])

        return {
            CamusTags.id: f"{patient_view_key}/{instant}",
            CamusTags.group: patient_view_key,
            CamusTags.img: img,
            CamusTags.gt: gt,
            CamusTags.frame_pos: frame_pos,
        }

    def _get_test_item(self, index: int) -> PatientData:
        """Fetches data required for inference on a test item, i.e. a patient.

        Args:
            index: Index of the test sample in the test set's ``self.item_list``.

        Returns:
            Data related a to a test item, i.e. a patient.
        """
        with h5py.File(self.root, "r") as dataset:
            patient_data = PatientData(id=self.item_list[index])
            for view in dataset[self.item_list[index]]:
                patient_view_key = f"{self.item_list[index]}/{view}"

                # Collect and process data
                proc_imgs, proc_gts, gts = Camus._get_data(
                    dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc, CamusTags.gt
                )
                proc_gts = self._process_target_data(proc_gts)
                gts = self._process_target_data(gts)

                # Collect metadata
                info, clinically_important_instants = Camus._get_metadata(
                    dataset, patient_view_key, CamusTags.info, CamusTags.instants
                )
                instants = {
                    instant: Camus._get_metadata(dataset, patient_view_key, instant)
                    for instant in clinically_important_instants
                }

                # If we do not use the whole sequence
                if self.dataset_with_sequence and not self.use_sequence:

                    # Only keep clinically important instants
                    instant_indices = list(instants.values())
                    proc_imgs = proc_imgs[instant_indices]
                    proc_gts = proc_gts[instant_indices]
                    gts = gts[instant_indices]

                    # Update indices of clinically important instants to match the new slicing of the sequences
                    instants = {instant: idx for idx, instant in enumerate(instants)}

                # Transform arrays to tensor
                proc_imgs_tensor = torch.stack([self._base_transform(proc_img) for proc_img in proc_imgs])
                proc_gts_tensor = torch.stack([self._base_target_transform(proc_gt) for proc_gt in proc_gts])

                # Extract metadata concerning the registering applied
                registering_parameters = None
                if self.registered_dataset:
                    registering_parameters = {
                        reg_step: Camus._get_metadata(dataset, patient_view_key, reg_step)
                        for reg_step in CamusRegisteringTransformer.registering_steps
                    }

                # Compute attributes for the sequence
                attrs = {CamusTags.frame_pos: torch.linspace(0, 1, steps=len(proc_imgs)).unsqueeze(1)}

                patient_data.views[view] = ViewData(
                    img_proc=proc_imgs_tensor,
                    gt_proc=proc_gts_tensor,
                    gt=gts,
                    voxelspacing=info[6:9][::-1],
                    instants=instants,
                    attrs=attrs,
                    registering=registering_parameters,
                )

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
