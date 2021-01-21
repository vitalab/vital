import os
from pathlib import Path
from typing import Tuple, List, Callable, Dict

import albumentations as A
import h5py
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms, ToTensor
from vital.data.acdc.config import AcdcTags, image_size, Label, AcdcSubset
from vital.data.acdc.data_struct import PatientData
from vital.data.acdc.transforms import NormalizeSample
from vital.data.acdc.utils import centered_resize


class Acdc(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the ACDC dataset."""

    def __init__(self,
                 path: Path,
                 image_set: AcdcSubset,
                 use_da: bool = False,
                 predict: bool = False,
                 transform: Callable = None,
                 target_transform: Callable = None):
        """
        Args:
            path: Path to the HDF5 dataset.
            image_set: select the subset of images to use from the enumeration.
            use_da: If True, data augmentation is applied when in train/validation mode.
            predict: whether to receive the data in a format fit for inference (``True``) or training (``False``).
            transform: a function/transform that takes in a numpy array and returns a transformed version.
            target_transform: a function/transform that takes in the target and transforms it.
        """
        transform = transforms.Compose([ToTensor(), NormalizeSample()]) \
            if not transform else transforms.Compose([transform, ToTensor()])
        target_transform = transforms.Compose([ToTensor()]) \
            if not target_transform else transforms.Compose([target_transform, ToTensor()])

        if use_da and image_set is AcdcSubset.TRAIN:
            self.da_transforms = A.Compose([A.Rotate(limit=60)])
        else:
            self.da_transforms = None

        super().__init__(path, transform=transform, target_transform=target_transform)

        self.image_set = image_set.value

        # TODO change dataset creation
        # with h5py.File(self.data_params.dataset, 'r') as f:
        #     self.registered_dataset = f.attrs[DataTags.registered]

        # Determine whether to return data in a format suitable for training or inference
        if predict:
            self.item_list = self._get_patient_paths()
            self.getter = self._get_test_item
        else:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_train_item

    def __getitem__(self, index):
        return self.getter(index)

    def __len__(self):
        return len(self.item_list)

    def get_num_classes(self) -> int:
        return len(list(Label))

    def _get_patient_paths(self) -> List[str]:
        """ Lists paths to the patients, from the requested ``image_set``, inside the HDF5 file.

        Returns:
            paths to the patients, from the requested ``image_set``, inside the HDF5 file.
        """
        with h5py.File(self.root, 'r') as dataset:
            patient_paths = [f'{self.image_set}/{patient_id}'
                             for patient_id in dataset[self.image_set].keys()]
        return patient_paths

    def _get_instant_paths(self) -> List[Tuple[str, int]]:
        """ Lists paths to the instants, from the requested ``image_set``, inside the HDF5 file.

        Returns:
            paths to the instants, from the requested ``image_set``, inside the HDF5 file.
        """

        image_paths = []
        patient_paths = self._get_patient_paths()
        with h5py.File(self.root, 'r') as dataset:
            for patient_path in patient_paths:
                patient = dataset[patient_path]
                for instant in range(patient[AcdcTags.gt].shape[0]):
                    image_paths.append((f'{patient_path}', instant))

        return image_paths

    def _get_train_item(self, index: int) -> Dict[str, torch.Tensor]:
        """ Fetches data required for training on a train/val item (single image/groundtruth pair).

        Args:
            index: index of the train/val sample in the train/val set's ``item_list``.

        Returns:
            data for training on a train/val item.
        """
        set_patient_key, instant = self.item_list[index]

        with h5py.File(self.root, 'r') as dataset:
            # Collect and process data
            patient_imgs, patient_gts = self._get_data(dataset, set_patient_key, AcdcTags.img, AcdcTags.gt)

            img = patient_imgs[instant]
            gt = patient_gts[instant]

            img, gt = self.center(img=img, gt=gt, output_shape=(image_size, image_size))

            voxel, = Acdc._get_metadata(dataset, set_patient_key, AcdcTags.voxel_spacing)

        # Get slice index
        slice_index = self.get_normalized_slice(patient_gts, instant, image_size)

        # Data augmentation transforms applied before Normalization and ToTensor
        if self.da_transforms:
            transformed = self.da_transforms(image=img, mask=gt)
            img = transformed["image"]
            gt = transformed["mask"]

        img = self.transform(img)
        gt = self.target_transform(gt)

        gt = gt.argmax(0)

        d = {AcdcTags.img: img,
             AcdcTags.gt: gt,
             AcdcTags.slice_index: slice_index,
             AcdcTags.voxel_spacing: voxel[:2],
             AcdcTags.id: set_patient_key + '_' + str(instant)
             }

        return d

    def _get_test_item(self, index: int) -> PatientData:
        """ Fetches data required for inference on a test item (whole patient).

        Args:
            index: index of the test sample in the test set's ``item_list``.

        Returns:
            data for inference on a test item.
        """

        with h5py.File(self.root, 'r') as dataset:
            set_patient_key = self.item_list[index]

            # Collect and process data
            imgs, gts = Acdc._get_data(dataset, set_patient_key, AcdcTags.img, AcdcTags.gt)

            imgs, gts = self.center(img=imgs, gt=gts, output_shape=(image_size, image_size))

            # Transform arrays to tensor
            imgs = torch.stack([self.transform(img) for img in imgs])
            gts = torch.stack([self.target_transform(gt) for gt in gts])

            gts = gts.argmax(1)

            voxel, = Acdc._get_metadata(dataset, set_patient_key, AcdcTags.voxel_spacing)

        return PatientData(id=os.path.basename(set_patient_key),
                           img=imgs,
                           gt=gts,
                           voxelspacing=voxel)

    @staticmethod
    def _get_data(file: h5py.File, set_patient_view_key: str, *data_tags: str) -> List[np.ndarray]:
        """ Fetches the requested data for a specific set/patient/view dataset from the HDF5 file.

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
    def _get_metadata(file: h5py.File, set_patient_key: str, *metadata_tags: str) -> List[np.ndarray]:
        """ Fetches the requested metadata for a specific set/patient/view dataset from the HDF5 file.

        Args:
            file: the HDF5 dataset file.
            set_patient_view_key: the `set/patient/view` access path of the desired view group.
            *metadata_tags: names of attributes to fetch from the view.

        Returns:
            attribute values for each tag passed in the parameters.
        """
        set_patient_view = file[set_patient_key]
        return [set_patient_view.attrs[attr_tag] for attr_tag in metadata_tags]

    @staticmethod
    def get_normalized_slice(gt: np.ndarray, instant: int, image_size: int) -> float:
        """
            Get the normalized index of the instant
        Args:
            gt (np.ndarray): Full patient segmentation map (N, H, W, K)
            instant (int): Index of the slice in the full segmentation map

        Returns:
            Normalize slice index between 0 and 1
        """
        number_slices = gt.shape[0]
        slice_index = int(instant)

        if np.sum(gt[0, :, :, 0]) != image_size * image_size:  # if first slice is not 0, add i virtually
            slice_index += 1
            number_slices += 1

        normalized_slice = slice_index / number_slices

        if np.sum(gt[:, :, 0]) == image_size * image_size:
            normalized_slice = 0

        return normalized_slice

    @staticmethod
    def center(img, gt, output_shape):
        img = centered_resize(img, output_shape)
        gt = centered_resize(gt, output_shape)

        summed = np.clip(gt[..., 1:].sum(axis=-1), 0, 1)
        gt[..., 0] = np.abs(1 - summed)

        return img, gt

    @staticmethod
    def get_voxel_spaces(dataset):
        """
            Get array of 2D voxel spacing for given dataset.
        Args:
            dataset (Dataset): Dataset from which to extract voxelspacing

        Returns:
            np.ndarry of 2D voxel spacings
        """
        return np.array([sample[AcdcTags.voxel_spacing][0:2] for sample in dataset])


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from argparse import ArgumentParser

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--use_da", action='store_true')
    params = args.parse_args()

    ds = Acdc(Path(params.path), image_set=AcdcSubset.TRAIN, predict=False, use_da=params.use_da)

    sample = ds[2] # ds[random.randint(0, len(ds) - 1)]

    img = sample[AcdcTags.img].squeeze()
    gt = sample[AcdcTags.gt]

    print("Image shape: {}".format(img.shape))
    print("GT shape: {}".format(gt.shape))
    print("Voxel_spacing: {}".format(sample[AcdcTags.voxel_spacing]))
    print("Slice index: {}".format(sample[AcdcTags.slice_index]))
    print("ID: {}".format(sample[AcdcTags.id]))

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(gt)
    plt.show(block=False)

    plt.figure(2)
    plt.imshow(img, cmap='gray')
    plt.imshow(gt, alpha=0.2)
    plt.show()
