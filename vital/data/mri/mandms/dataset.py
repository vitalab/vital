from pathlib import Path
from typing import Callable, Dict, Literal, List

import h5py
import numpy as np
import torch

from vital.data.config import Subset
from vital.data.mri.config import MRITags
from vital.data.mri.dataset import ShortAxisMRI, visualize_dataset


class MandM(ShortAxisMRI):
    """Implementation of torchvision's ``VisionDataset`` for the M&Ms dataset."""

    VENDORS = ['Canon', 'GE', 'Philips', 'Siemens']

    def __init__(
        self,
        path: Path,
        image_set: Subset,
        use_da: bool = False,
        predict: bool = False,
        transform: Callable = None,
        target_transform: Callable = None,
        vendors: List[str] = None
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
        self.vendors = vendors or self.VENDORS
        assert set(self.VENDORS).issuperset(set(self.vendors))

        super().__init__(path, image_set, use_da, predict, transform, target_transform)

        assert not (predict and self.image_set == Subset.UNLABELED.value)
        if self.image_set == Subset.UNLABELED.value:
            self.item_list = self._get_instant_paths()
            self.getter = self._get_unlabled_item

    def list_groups(self, level: Literal["patient", "instant"] = "instant") -> List[str]:
        """Lists the paths of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.

        Args:
            level: Hierarchical level at which to group data samples.
                - 'patient': all the data from the same patient is associated to a unique ID.
                - 'instant': all the data from the same instant of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.
        """
        groups = super().list_groups(level)

        accepted_vendor_patients = []

        with h5py.File(self.root, "r") as dataset:
            for vendor in dataset['vendors']:
                if vendor in self.vendors:
                        accepted_vendor_patients.extend(dataset['vendors'][vendor])

        accepted_vendor_patients = [x.decode("utf-8") for x in accepted_vendor_patients]

        print(accepted_vendor_patients)

        groups = [x for x in groups if x.split('/')[1] in accepted_vendor_patients]

        return groups

    @staticmethod
    def correct_gt_labels(gt: np.ndarray) -> np.ndarray:
        """Correct the labels values for different datasets.

        Args:
            gt: categorical segmentation map either  ([N], H, W) or (H, W)

        Returns:
            gt with corrected class values
        """
        # FLip RV and LV class labels
        copy = np.copy(gt)
        copy[gt == 1] = 3
        copy[gt == 3] = 1
        return copy

    def _get_unlabled_item(self, index: int) -> Dict[str, torch.Tensor]:
        """Fetches data required for training on a labeled item (single image without groundtruth).

        Args:
            index: index of the train/val sample in the train/val set's ``item_list``.

        Returns:
            data for training on a train/val item.
        """
        set_patient_instant_key, slice = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            # Collect and process data
            (patient_imgs,) = self._get_data(dataset, set_patient_instant_key, MRITags.img)

            img = patient_imgs[slice]

            (voxel,) = ShortAxisMRI._get_metadata(dataset, set_patient_instant_key, MRITags.voxel_spacing)

        # Data augmentation transforms applied before Normalization and ToTensor as it is done on np.ndarray
        if self.da_transforms:
            transformed = self.da_transforms(image=img)
            img = transformed["image"]

        img = self.transform(img)

        d = {
            MRITags.img: img,
            MRITags.gt: None,
            MRITags.slice_index: None,
            MRITags.voxel_spacing: voxel[:2],
            MRITags.id: set_patient_instant_key + "_" + str(slice),
        }

        return d


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--use_da", action="store_true")
    args.add_argument("--predict", action="store_true")
    params = args.parse_args()

    ds = MandM(Path(params.path), image_set=Subset.TRAIN, predict=params.predict, use_da=params.use_da,
               vendors=['Canon', 'GE', 'Philips'])

    print(ds.list_groups('instant'))
    print(ds.list_groups('patient'))

    print(len(ds))

    visualize_dataset(ds, params.predict)
