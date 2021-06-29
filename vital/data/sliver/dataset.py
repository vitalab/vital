from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import albumentations as A

from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor, transforms

from vital.data.config import Subset
from vital.data.sliver.config import SLiverTags
from vital.data.transforms import NormalizeSample, SegmentationToTensor, RescaleIntensity


class SliverDataset(VisionDataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(
        self,
        path: Path,
        image_set: Subset,
        use_da: bool = False,
        predict: bool = False,
        transform: Callable = None,
        target_transform: Callable = None,
    ):
        """
        Args:
        """
        transform = (
            transforms.Compose([ToTensor(), NormalizeSample(), RescaleIntensity()])
            if not transform
            else transforms.Compose([transform, ToTensor(), RescaleIntensity()])
        )
        target_transform = (
            transforms.Compose([SegmentationToTensor(), RescaleIntensity()])
            if not target_transform
            else transforms.Compose([target_transform, SegmentationToTensor(), RescaleIntensity()])
        )

        if use_da and image_set is Subset.TRAIN:
            self.da_transforms = A.Compose([A.HorizontalFlip()])
        else:
            self.da_transforms = None

        super().__init__(path, transform=transform, target_transform=target_transform)

        self.image_set = image_set.value

        self.file_path = path
        self.predict = predict
        self.data = self._load_data()

        with h5py.File(self.file_path, "r") as f:
            self.item_list = list(f[self.image_set].keys())

    def _load_data(self):
        """List of images from the HDF5 file and save them into a set list.

        :return
            list containing path to files for the corresponding set
        """

        def get_set_list(set_key, file, predict=False):
            set_list = []
            f_set = file[set_key]
            for key in list(f_set.keys()):
                patient = f_set[key]
                img = patient[SLiverTags.img]
                gt = patient[SLiverTags.gt]

                assert img.shape[0] == gt.shape[0]

                for i in range(img.shape[0]):
                    k = "{}/{}".format(set_key, key)
                    set_list.append((k, i))
            return set_list

        with h5py.File(self.file_path, "r") as f:
            set_list = get_set_list(self.image_set, f, self.predict)
        return set_list

    def __getitem__(self, index):
        """This method loads, transforms and returns slice corresponding to the corresponding index.
        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)

        """
        return self._getitem_2d(index)

    def __len__(self):
        """
        return the length of the dataset
        """
        d_size = int(np.floor(len(self.data)))
        return d_size

    def _getitem_2d(self, index):
        key, position = self.data[index]

        with h5py.File(self.file_path, "r") as hdf_handle:
            img, gt = self.get_data_slice(key, position, hdf_handle)
            voxel_size = hdf_handle[key].attrs.get("voxel_size", [1, 1, 1])
            voxel_size = self.fix_voxel_size(voxel_size)
        # Prepare shape and type of input data

        img = img.astype(np.float32)
        gt = gt.astype(np.uint8)

        # Data augmentation transforms applied before Normalization and ToTensor as it is done on np.ndarray
        if self.da_transforms:
            transformed = self.da_transforms(image=img, mask=gt)
            img = transformed["image"]
            gt = transformed["mask"]

        img = self.transform(img)
        gt = self.target_transform(gt).squeeze()[None, ...].float()

        d = {
            SLiverTags.img: img,
            SLiverTags.gt: gt,
            SLiverTags.voxel_spacing: voxel_size[:2],
            SLiverTags.id: f"{key}_{position}",
        }

        return d

    def fix_voxel_size(self, voxel_size):
        z_idx = np.argmax(voxel_size)
        if z_idx == 2:
            voxel_size = np.array([voxel_size[z_idx], voxel_size[0], voxel_size[1]])
        return voxel_size

    @staticmethod
    def get_data_slice(key, position, file):
        """
        Return one slice from the hdf5 file
        Args:
            key: key corresponding to each patient
            position: image and ground truth positions into patient's images
            file: the hdf5 dataset
        :return
            tuple corresponding to a slice and its corresponding ground truth
        """
        img = file["{}/{}".format(key, SLiverTags.img)][int(position)]
        gt = file["{}/{}".format(key, SLiverTags.gt)][int(position)]

        return img, gt

    @staticmethod
    def get_data_volume(key, file):
        """
        Return one slice from the hdf5 file
        Args:
            key: key corresponding to each patient
            file: the hdf5 dataset
        :return
            tuple corresponding to a slice and its corresponding ground truth
        """
        img = file["{}/{}".format(key, SLiverTags.img)]
        gt = file["{}/{}".format(key, SLiverTags.gt)]
        return img, gt


"""
This script can be run to test and visualize the data from the dataset.
"""
if __name__ == "__main__":
    from argparse import ArgumentParser
    from matplotlib import pyplot as plt
    import random

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--use_da", action="store_true")
    params = args.parse_args()

    ds = SliverDataset(Path(params.path), image_set=Subset.TRAIN, use_da=params.use_da)

    sample = ds[random.randint(0, len(ds) - 1)]
    img = sample[SLiverTags.img].squeeze()
    gt = sample[SLiverTags.gt].squeeze()
    print("Image shape: {}".format(img.shape))
    print("GT shape: {}".format(gt.shape))
    print("Voxel_spacing: {}".format(sample[SLiverTags.voxel_spacing]))

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(gt)
    plt.show(block=False)

    # plt.figure(2)
    # plt.imshow(img, cmap="gray")
    # plt.imshow(gt, alpha=0.2)

    plt.show()
