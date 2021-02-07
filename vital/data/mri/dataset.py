from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple

import albumentations as A
import h5py
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor, transforms

from vital.data.config import Subset
from vital.data.mri.acdc.utils.acdc import AcdcRegisteringTransformer
from vital.data.mri.config import Instant, MRITags, image_size
from vital.data.mri.data_struct import InstantData, PatientData
from vital.data.mri.transforms import NormalizeSample, SegmentationToTensor
from vital.utils.decorators import squeeze


class ShortAxisMRI(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for short axis MRI segmentation datasets."""

    def __init__(
        self,
        path: Path,
        image_set: Subset,
        use_da: bool = False,
        predict: bool = False,
        transform: Callable = None,
        target_transform: Callable = None,
    ):  # noqa: D205,D212,D415
        """
        Args:
            path: Path to the HDF5 dataset.
            image_set: select the subset of images to use from the enumeration.
            use_da: If True, data augmentation is applied when in train/validation mode.
            predict: whether to receive the data in a format fit for inference (``True``) or training (``False``).
            transform: a function/transform that takes in a numpy array and returns a transformed version.
            target_transform: a function/transform that takes in the target and transforms it.
        """
        transform = (
            transforms.Compose([ToTensor(), NormalizeSample()])
            if not transform
            else transforms.Compose([transform, ToTensor()])
        )
        target_transform = (
            transforms.Compose([SegmentationToTensor()])
            if not target_transform
            else transforms.Compose([target_transform, SegmentationToTensor()])
        )

        if use_da and image_set is Subset.TRAIN:
            self.da_transforms = A.Compose([A.Rotate(limit=60)])
        else:
            self.da_transforms = None

        super().__init__(path, transform=transform, target_transform=target_transform)

        self.image_set = image_set.value

        with h5py.File(path, "r") as f:
            if MRITags.registered in f.attrs.keys():
                self.registered_dataset = f.attrs[MRITags.registered]
            else:
                self.registered_dataset = False

        # Determine whether to return data in a format suitable for training or inference
        if predict:
            self.item_list = self.list_groups(level="patient")
            self.getter = self._get_test_item
        else:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_train_item

    def __getitem__(self, index):
        """Fetches an item, whose structure depends on the ``predict`` value, from the internal list of items.

        Notes:
            - When in ``predict`` mode (i.e. for test-time inference), an item corresponds to full patient (ES and ED)
              images and groundtruth segmentations for a patient.
            - When not in ``predict`` mode (i.e. during training), an item corresponds to an image/segmentation pair for
              a single slice.

        Args:
            index: Index of the item to fetch from the internal sequence of items.

        Returns:
            Item from the internal list at position ``index``.
        """
        return self.getter(index)

    def __len__(self):  # noqa: D105
        return len(self.item_list)

    def list_groups(self, level: Literal["patient", "instant"] = "instant") -> List[str]:
        """Lists the paths of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'instant': all the data from the same instant of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.
        """
        with h5py.File(self.root, "r") as dataset:
            # List the patients
            groups = [f"{self.image_set}/{patient_id}" for patient_id in dataset[self.image_set].keys()]

            if level == "instant":
                groups = [f"{patient}/{instant}" for patient in groups for instant in dataset[patient].keys()]

        return groups

    def _get_instant_paths(self) -> List[Tuple[str, int]]:
        """Lists paths to the instants, from the requested ``image_set``, inside the HDF5 file.

        Returns:
            paths to the instants, from the requested ``image_set``, inside the HDF5 file.
        """
        image_paths = []
        instant_paths = self.list_groups(level="instant")
        with h5py.File(self.root, "r") as dataset:
            for instant_path in instant_paths:
                num_slices = len(dataset[instant_path][MRITags.img])
                image_paths.extend((instant_path, slice) for slice in range(num_slices))

        return image_paths

    def _get_train_item(self, index: int) -> Dict[str, torch.Tensor]:
        """Fetches data required for training on a train/val item (single image/groundtruth pair).

        Args:
            index: index of the train/val sample in the train/val set's ``item_list``.

        Returns:
            data for training on a train/val item.
        """
        set_patient_instant_key, slice = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            # Collect and process data
            patient_imgs, patient_gts = self._get_data(dataset, set_patient_instant_key, MRITags.img, MRITags.gt)

            img = patient_imgs[slice]
            gt = patient_gts[slice]

            voxel = ShortAxisMRI._get_metadata(dataset, set_patient_instant_key, MRITags.voxel_spacing)

        # Get slice index
        slice_index = self.get_normalized_slice(patient_gts, slice, image_size)

        # Data augmentation transforms applied before Normalization and ToTensor as it is done on np.ndarray
        if self.da_transforms:
            transformed = self.da_transforms(image=img, mask=gt)
            img = transformed["image"]
            gt = transformed["mask"]

        img = self.transform(img)
        gt = self.target_transform(gt).squeeze()

        d = {
            MRITags.img: img,
            MRITags.gt: gt,
            MRITags.slice_index: slice_index,
            MRITags.voxel_spacing: voxel[:2],
            MRITags.id: f"{set_patient_instant_key}_{slice}",
        }

        return d

    def _get_test_item(self, index: int) -> PatientData:
        """Fetches data required for inference on a test item (whole patient).

        Args:
            index: index of the test sample in the test set's ``item_list``.

        Returns:
            data for inference on a test item.
        """
        with h5py.File(self.root, "r") as dataset:
            patient_key = self.item_list[index]
            patient_data = PatientData(id=patient_key)

            for instant in dataset[patient_key]:
                patient_instant_key = f"{patient_key}/{instant}"

                # Collect and process data
                imgs, gts = ShortAxisMRI._get_data(dataset, patient_instant_key, MRITags.img, MRITags.gt)

                # Transform arrays to tensor
                imgs = torch.stack([self.transform(img) for img in imgs])
                gts = torch.stack([self.target_transform(gt) for gt in gts]).squeeze()

                # Extract metadata concerning the registering applied
                registering_parameters = None
                if self.registered_dataset:
                    registering_parameters = {
                        reg_step: ShortAxisMRI._get_metadata(dataset, patient_instant_key, reg_step)
                        for reg_step in AcdcRegisteringTransformer.registering_steps
                    }

                voxel = ShortAxisMRI._get_metadata(dataset, patient_instant_key, MRITags.voxel_spacing)

                patient_data.instants[instant] = InstantData(
                    img=imgs, gt=gts, registering=registering_parameters, voxelspacing=voxel
                )

        return patient_data

    @staticmethod
    @squeeze
    def _get_data(file: h5py.File, set_patient_instant_key: str, *data_tags: str) -> List[np.ndarray]:
        """Fetches the requested data for a specific set/patient/instant dataset from the HDF5 file.

        Args:
            file: the HDF5 dataset file.
            set_patient_instant_key: the `set/patient/instant` access path of the desired instant group.
            *data_tags: names of the datasets to fetch from the instant.

        Returns:
            Dataset content for each tag passed in the parameters.
        """
        set_patient_instant = file[set_patient_instant_key]
        return [set_patient_instant[data_tag][()] for data_tag in data_tags]

    @staticmethod
    @squeeze
    def _get_metadata(file: h5py.File, set_patient_instant_key: str, *metadata_tags: str) -> List[np.ndarray]:
        """Fetches the requested metadata for a specific set/patient/instant dataset from the HDF5 file.

        Args:
            file: the HDF5 dataset file.
            set_patient_instant_key: the `set/patient/instant` access path of the desired instant group.
            *metadata_tags: names of attributes to fetch from the instant.

        Returns:
            Attribute values for each tag passed in the parameters.
        """
        set_patient = file[set_patient_instant_key]
        return [set_patient.attrs[attr_tag] for attr_tag in metadata_tags]

    @staticmethod
    def get_normalized_slice(gt: np.ndarray, instant: int, image_size: int) -> float:
        """Get the normalized index of the instant.

        Args:
            gt: Full patient segmentation map (N, H, W)
            instant: Index of the slice in the full segmentation map
            image_size: size of the image

        Returns:
            Normalize slice index between 0 and 1
        """
        number_slices = gt.shape[0]
        slice_index = int(instant)

        if np.sum(np.equal(gt[slice_index], 0)) == image_size * image_size:
            return 0  # If slice is empty, return 0

        if np.sum(np.equal(gt[0, :, :], 0)) != image_size * image_size:  # if first slice is not 0, add i virtually
            slice_index += 1
            number_slices += 1

        return slice_index / number_slices

    @staticmethod
    def get_voxel_spaces(dataset):
        """Get array of 2D voxel spacing for given dataset.

        Args:
            dataset (Dataset): Dataset from which to extract voxelspacing

        Returns:
            np.ndarry of 2D voxel spacings
        """
        return np.array([sample[MRITags.voxel_spacing][0:2] for sample in dataset])


def visualize_dataset(dataset: ShortAxisMRI, predict: bool):
    """Visualize the datasets images and gt.

    Args:
        dataset: dataset from which to get samples
        predict: whether the dataset is in predict mode
    """
    import random

    from matplotlib import pyplot as plt

    if predict:
        patient = dataset[random.randint(0, len(dataset) - 1)]
        instant = patient.instants[Instant.ED.value]
        img = instant.img
        gt = instant.gt
        print("Image shape: {}".format(img.shape))
        print("GT shape: {}".format(gt.shape))
        print("Voxel_spacing: {}".format(instant.voxelspacing.shape))
        print("ID: {}".format(patient.id))

        slice = random.randint(0, len(img) - 1)
        img = img[slice].squeeze()
        gt = gt[slice]

    else:
        sample = dataset[random.randint(0, len(dataset) - 1)]
        img = sample[MRITags.img].squeeze()
        gt = sample[MRITags.gt]
        print("Image shape: {}".format(img.shape))
        print("GT shape: {}".format(gt.shape))
        print("Voxel_spacing: {}".format(sample[MRITags.voxel_spacing]))
        print("Slice index: {}".format(sample[MRITags.slice_index]))

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(gt)
    plt.show(block=False)

    plt.figure(2)
    plt.imshow(img, cmap="gray")
    plt.imshow(gt, alpha=0.2)
    plt.show()
