import argparse
import os

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from vital.data.mri.config import Instant, Label, MRITags, image_size
from vital.data.mri.mandms.dataset import Subset
from vital.data.mri.utils import centered_resize, centered_resize_gt, to_onehot

img_name_format = "{}_sa.nii.gz"
gt_name_format = "{}_sa_gt.nii.gz"
csv_info_filename = "201014_M&Ms_Dataset_Information_-_opendataset.csv"


def generate_dataset(path: str, name: str):
    """Generate the h5 dataset.

    Args:
        path: path to the raw data.
        name: Name of the output file.
    """
    dataset_info = pd.read_csv(os.path.join(path, csv_info_filename), index_col="External code")

    vendors = list(set(dataset_info["VendorName"]))
    vendors = {vendor: list(dataset_info.loc[dataset_info["VendorName"] == vendor].index) for vendor in vendors}

    with h5py.File(name, "w") as h5f:
        # List all the samples by vendor.
        vendor_group = h5f.create_group("vendors")
        for k, v in vendors.items():
            vendor_group.create_dataset(k, data=np.array(v, dtype="S"))

        # Create training sets
        print("Training/Labeled...")
        labeled_train_group = h5f.create_group(Subset.TRAIN.value)
        generate_set(os.path.join(path, "Training/Labeled"), dataset_info, labeled_train_group)

        print("Training/Unlabeled...")
        unlabeled_train_group = h5f.create_group(Subset.UNLABELED.value)
        generate_set(os.path.join(path, "Training/Unlabeled"), dataset_info, unlabeled_train_group, labeled=False)

        print("Validation...")
        val_group = h5f.create_group(Subset.VAL.value)
        generate_set(os.path.join(path, "Validation"), dataset_info, val_group)

        # print("Testing...")
        # test_group = h5f.create_group(Subset.TEST.value)
        # generate_set(os.path.join(path, 'Testing'), dataset_info, test_group)


def generate_set(path: str, dataset_info: pd.DataFrame, group: h5py.Group, labeled: bool = True):
    """Generate a group for one set of the dataset.

    Args:
        path: Path to the raw data.
        dataset_info: dataframe containing info about samples (ie. index of ES and ED instant)
        group: group for which to save data
        labeled: whether the set is labeled
    """
    patients_ids = os.listdir(path)

    for patient_id in tqdm(patients_ids):
        patient_dir = os.path.join(path, patient_id)
        patient_info = dataset_info.loc[patient_id]

        es_idx = patient_info["ES"]
        ed_idx = patient_info["ED"]

        patient_group = group.create_group(patient_id)

        ni_img = nib.load(os.path.join(patient_dir, img_name_format.format(patient_id)))
        img = ni_img.get_fdata().astype(np.float32)
        img = img.transpose(2, 0, 1, 3)[..., np.newaxis]  # (x, y, z, t) -> (z, x, y, t)
        voxel_size = ni_img.header.get_zooms()

        ed_instant = patient_group.create_group(Instant.ED.value)
        ed_instant.attrs[MRITags.voxel_spacing] = voxel_size
        ed_img = centered_resize(img[:, :, :, ed_idx], (image_size, image_size))
        ed_instant.create_dataset(MRITags.img, data=ed_img)

        es_instant = patient_group.create_group(Instant.ES.value)
        es_instant.attrs[MRITags.voxel_spacing] = voxel_size
        es_img = centered_resize(img[:, :, :, es_idx], (image_size, image_size))
        es_instant.create_dataset(MRITags.img, data=es_img)

        if labeled:
            ni_gt = nib.load(os.path.join(patient_dir, gt_name_format.format(patient_id)))
            gt = ni_gt.get_fdata().astype(np.float32)
            gt = gt.transpose(2, 0, 1, 3)[..., np.newaxis]  # (x, y, z, t) -> (z, x, y, t)
            gt = to_onehot(gt, Label.count()).astype(np.uint8)

            # Center and resize the gt and convert to categorical format.
            gt_ed = centered_resize_gt(gt[:, :, :, ed_idx], (image_size, image_size)).argmax(-1)
            gt_es = centered_resize_gt(gt[:, :, :, es_idx], (image_size, image_size)).argmax(-1)

            ed_instant.create_dataset(MRITags.gt, data=gt_ed)
            es_instant.create_dataset(MRITags.gt, data=gt_es)


def main():
    """Main function where we define the argument for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Script to create the ACDC dataset "
            "hdf5 file from the directory. "
            "The given directory need to have two "
            "directory inside, 'training' and 'testing'."
        )
    )
    parser.add_argument("--path", type=str, required=True, help="Path of the ACDC dataset.")
    parser.add_argument("--name", type=str, required=True, help="Name of the generated hdf5 file.")

    args = parser.parse_args()

    generate_dataset(args.path, args.name)


if __name__ == "__main__":
    main()
