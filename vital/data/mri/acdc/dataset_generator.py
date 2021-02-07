import argparse
import os
from glob import glob
from os.path import basename
from typing import List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
from natsort import natsorted
from scipy import ndimage
from scipy.ndimage.interpolation import rotate

from vital.data.config import Subset
from vital.data.mri.acdc.utils.acdc import AcdcRegisteringTransformer
from vital.data.mri.config import Instant, Label, MRITags, image_size
from vital.data.mri.utils import centered_resize, centered_resize_gt, to_onehot

try:
    from tqdm import tqdm
except ImportError:
    print("The package 'tqdm' is not installed, " "if you want a progress bar install it.")

    def tqdm(iterable, *args, **kwargs):
        """Mcock function."""
        return iterable


PRIOR_SIZE = 100
PRIOR_HALF_SIZE = PRIOR_SIZE // 2

ROTATIONS = [-60, -45, -15, 0, 15, 45, 60]


def generate_list_directory(path: str):
    """Generate the list of nifti images from the path.

    Args:
        path: path to nifti images
    """
    if not os.path.exists(path):
        return []

    path = os.path.join(path, "*", "*")
    paths = natsorted(glob(path))
    paths = [path for path in paths if "Info" not in path]
    paths = [path for path in paths if path.find("_4d") < 0]
    return paths


def _mass_center(imgs: List[np.ndarray]):
    """Function to extract the center of masses of a 3D ground truth image.

    Args:
        imgs: images
    """
    centers = np.array([ndimage.measurements.center_of_mass(np.equal(img, 3)) for img in imgs])
    # Need to fix the Nan when the ground truth slice is a slice of zeros.
    # Set it to center 256 // 2 = 128
    centers[np.isnan(centers)] = 128
    return centers.astype(np.int16)


def _generate_centered_prob_map(image: np.ndarray, shape: np.ndarray, center: np.ndarray, label: int):
    """Function to extract the information from the ground truth image given the centers.

    Args:
        image: Numpy array of the ground truth.
        shape: Shape of the desired prior image.
        center: Numpy array of shape (slices, 2).
        label: Which label to extract from the ground truth.

    Returns:
        Array of shape (slices, 100, 100, 1)
    """
    image = np.equal(image, label)[..., None]
    res = np.zeros(shape)
    # Nearest neighbour slice index between the number of slice
    # of the image and the ground truth
    space = np.linspace(0, shape[0] - 1, num=image.shape[0]).astype(np.int32)
    for i, (s, c) in enumerate(zip(space, center)):
        res[s] += image[
            i,
            c[0] - PRIOR_HALF_SIZE : c[0] + PRIOR_HALF_SIZE,
            c[1] - PRIOR_HALF_SIZE : c[1] + PRIOR_HALF_SIZE,
        ]
    return res


def generate_probability_map(h5f: h5py.File, group: h5py.Group):
    """Generate the probability map from all unrotated training exemples.

    Args:
        h5f: Handle of the hdf5 file containing all the dataset.
        group: Group where to create the prior shape (train, valid or test) train should be the
            default.
    """
    patient_keys = [key for key in group.keys() if key.endswith("_0")]
    image_keys = []
    for k1 in patient_keys:
        for k2 in group[k1].keys():
            image_keys.append("{}/{}/{}".format(k1, k2, MRITags.gt))

    images = [group[k][:] for k in image_keys]
    images_center = [_mass_center(img) for img in images]

    prior_shape = np.array([15, PRIOR_SIZE, PRIOR_SIZE, 1])

    label0 = np.array(
        [
            _generate_centered_prob_map(np.copy(img), prior_shape, center, 0)
            for img, center in tqdm(zip(images, images_center), desc="Background", total=len(images))
        ]
    )
    label0 = label0.sum(axis=0)

    label1 = np.array(
        [
            _generate_centered_prob_map(np.copy(img), prior_shape, center, 1)
            for img, center in tqdm(zip(images, images_center), desc="Right ventricle", total=len(images))
        ]
    )
    label1 = label1.sum(axis=0)

    label2 = np.array(
        [
            _generate_centered_prob_map(np.copy(img), prior_shape, center, 2)
            for img, center in tqdm(zip(images, images_center), desc="Myocardium", total=len(images))
        ]
    )
    label2 = label2.sum(axis=0)

    label3 = np.array(
        [
            _generate_centered_prob_map(np.copy(img), prior_shape, center, 3)
            for img, center in tqdm(zip(images, images_center), desc="Left ventricle", total=len(images))
        ]
    )
    label3 = label3.sum(axis=0)

    p_img = np.concatenate((label0, label1, label2, label3), axis=-1)
    p_img /= p_img.sum(axis=-1, keepdims=True).astype(np.float32)
    h5f.create_dataset("prior", data=p_img[:, :, :, 1:])


def load_instant_data(img_path: str, gt_path: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load data, gt (when available) and voxel spacing from  nii file.

    Args:
        img_path: path to image nii file.
        gt_path: path to gt nii file.

    Returns:
        img, gt and voxel spacing for one instant.
    """
    ni_img = nib.load(img_path)

    img = ni_img.get_fdata().astype(np.float32)
    img = img.transpose(2, 0, 1)[..., np.newaxis]

    img = centered_resize(img, (image_size, image_size))

    voxel = ni_img.header.get_zooms()

    if gt_path:
        ni_img = nib.load(gt_path)
        gt = ni_img.get_fdata()
        gt = gt.transpose(2, 0, 1)[..., np.newaxis]

        gt = to_onehot(gt, Label.count()).astype(np.uint8)

        gt = centered_resize_gt(gt, (image_size, image_size))

        # Put data to categorical format.
        gt = gt.argmax(-1)

        return img, gt, voxel

    return img, None, voxel


def write_instant_group(
    patient_group: h5py.Group,
    group_name: str,
    img_data: np.ndarray,
    gt_data: np.ndarray,
    voxel: np.ndarray,
    rotation: float,
    registering_transformer: Optional[AcdcRegisteringTransformer],
):
    """Write an instant group to the patient group.

    Args:
        patient_group: patient group for which the instant is saved
        group_name: name of the instant group name
        img_data: Array containing the img data
        gt_data: Array containing the gt data
        voxel: Voxel size of the instant
        rotation: Rotation for data augmentation
        registering_transformer: Transformer used for registration
    """
    instant = patient_group.create_group(group_name)
    instant.attrs["voxel_size"] = voxel

    r_img = rotate(img_data, rotation, axes=(1, 2), reshape=False)
    r_img = np.clip(r_img, img_data.min(), img_data.max())
    r_img[np.isclose(r_img, 0.0)] = 0.0

    if registering_transformer is not None:
        registering_parameters, gt_data, r_img = registering_transformer.register_batch(gt_data, r_img)
        instant.attrs.update(registering_parameters)

    instant.create_dataset(MRITags.img, data=r_img)

    if gt_data is not None:
        r_img = rotate(gt_data, rotation, axes=(1, 2), output=np.uint8, reshape=False)
        instant.create_dataset(MRITags.gt, data=r_img)


def create_database_structure(
    group: h5py.Group,
    data_augmentation: bool,
    registering: bool,
    data_ed: str,
    gt_ed: str,
    data_es: str,
    gt_es: str,
    data_mid: Optional[str] = None,
    gt_mid: Optional[str] = None,
):
    """Create the dataset for the End-Systolic and End-Diastolic phases.

    If some data augmentation is involved we create also the rotation for each phase.

    Args:
        group: Group where we add each image by its name and the rotation associated to it.
        data_augmentation: Enable/Disable data augmentation.
        registering: Enable/Disable registering.
        data_ed: Path of the nifti diastolic MRI image.
        gt_ed: Path of the nifti diastolic MRI segmentation of data_ed.
        data_es: Path of the nifti systolic MRI image.
        gt_es: Path of the nifti systolic MRI segmentation of data_ed.
        data_mid: Path of the nifti MRI image at a time step between ED and ES.
        gt_mid: Path of the nifti MRI segmentation of data_mid.

    """
    p_name = data_ed.split(os.path.sep)[-2]

    ed_img, edg_img, ed_voxel = load_instant_data(data_ed, gt_ed)
    es_img, esg_img, es_voxel = load_instant_data(data_es, gt_es)

    if data_mid:
        mid_img, midg_img, mid_voxel = load_instant_data(data_mid, gt_mid)

    if data_augmentation:
        iterable = ROTATIONS
    else:
        iterable = [
            0,
        ]

    if registering:
        registering_transformer = AcdcRegisteringTransformer()
    else:
        registering_transformer = None

    for rot in iterable:
        name = f"{p_name}_{rot}" if len(iterable) > 1 else p_name
        patient = group.create_group(name)

        write_instant_group(patient, Instant.ED.value, ed_img, edg_img, ed_voxel, rot, registering_transformer)
        write_instant_group(patient, Instant.ES.value, es_img, esg_img, es_voxel, rot, registering_transformer)

        # Add mid-cycle data
        if data_mid:
            write_instant_group(patient, Instant.MID.value, mid_img, midg_img, mid_voxel, rot, registering_transformer)


def generate_dataset(path: str, name: str, data_augmentation: bool = False, registering: bool = False):
    """Function that generates each dataset, train, valid and test.

    Args:
        path: Path where we can find the images from the downloaded ACDC challenge.
        name: Name of the hdf5 file to generate.
        data_augmentation: Enable/Disable data augmentation.
        registering: Enable/Disable registering.

    Raises:
        ValueError if names don't match
    """
    if data_augmentation:
        print("Data augmentation enabled, rotation " "from -60 to 60 by step of 15.")
    if registering:
        print("Registering enabled, MRIs and groundtruths centered and rotated.")
    rng = np.random.RandomState(1337)

    # get training examples
    train_paths = generate_list_directory(os.path.join(path, "training"))
    # We have 4 path, path_ED, path_gt_ED, path_ES and path_gt_ES
    train_paths = np.array(list(zip(train_paths[0::4], train_paths[1::4], train_paths[2::4], train_paths[3::4])))

    # 20 is the number of patients per group
    patients_per_group = 20
    indexes = np.arange(patients_per_group)

    train_idxs = []
    valid_idxs = []
    # 5 is the number of groups
    for i in range(5):
        start = i * patients_per_group
        idxs = indexes + start
        rng.shuffle(idxs)
        t_idxs = idxs[: int(indexes.shape[0] * 0.75)]
        v_idxs = idxs[int(indexes.shape[0] * 0.75) :]
        train_idxs.append(t_idxs)
        valid_idxs.append(v_idxs)

    train_idxs = np.array(train_idxs).flatten()
    valid_idxs = np.array(valid_idxs).flatten()
    valid_paths = train_paths[valid_idxs]
    train_paths = train_paths[train_idxs]

    # get testing examples
    if os.path.exists(os.path.join(path, "testing_with_gt_mid")):
        test_paths = generate_list_directory(os.path.join(path, "testing_with_gt_mid"))
        test_paths = np.array(
            list(
                zip(
                    test_paths[0::6],
                    test_paths[1::6],
                    test_paths[2::6],
                    test_paths[3::6],
                    test_paths[4::6],
                    test_paths[5::6],
                )
            )
        )
    else:
        test_paths = generate_list_directory(os.path.join(path, "testing_with_gt"))
        test_paths = np.array(list(zip(test_paths[0::4], test_paths[1::4], test_paths[2::4], test_paths[3::4])))
        test_paths = np.array([np.insert(i, 2, [None, None]).tolist() for i in test_paths])

    with h5py.File(name, "w") as h5f:

        h5f.attrs[MRITags.registered] = registering

        # Training samples ###
        group = h5f.create_group(Subset.TRAIN.value)
        for p_ed, g_ed, p_es, g_es in tqdm(train_paths, desc="Training"):
            # Find missmatch in the zip
            if p_ed != g_ed.replace("_gt", "") or p_es != g_es.replace("_gt", ""):
                raise ValueError(
                    (
                        "File name don't match: ",
                        "{} instead of {}, ".format(p_ed, g_ed),
                        "{} instead of {}.".format(p_es, g_es),
                    )
                )

            create_database_structure(group, data_augmentation, registering, p_ed, g_ed, p_es, g_es)

        # Generate the probability map from the ground truth training examples
        generate_probability_map(h5f, group)

        # Validation samples ###
        group = h5f.create_group(Subset.VAL.value)
        for p_ed, g_ed, p_es, g_es in tqdm(valid_paths, desc="Validation"):
            # Find missmatch in the zip
            if p_ed != g_ed.replace("_gt", "") or p_es != g_es.replace("_gt", ""):
                raise ValueError(
                    (
                        "File name don't match: ",
                        "{} instead of {}, ".format(p_ed, g_ed),
                        "{} instead of {}.".format(p_es, g_es),
                    )
                )

            create_database_structure(group, False, registering, p_ed, g_ed, p_es, g_es)

        # Testing samples ###
        group = h5f.create_group(Subset.TEST.value)
        for p_ed, g_ed, p_mid, g_mid, p_es, g_es in tqdm(test_paths, desc="Testing"):
            p_mid = None if p_mid == "None" else p_mid
            g_mid = None if g_mid == "None" else g_mid
            # Find missmatch in the zip
            if basename(p_ed) != basename(g_ed).replace("_gt", "") or basename(p_es) != basename(g_es).replace(
                "_gt", ""
            ):
                raise ValueError(
                    (
                        "File name don't match: ",
                        "{} instead of {}, ".format(p_ed, g_ed),
                        "{} instead of {}.".format(p_es, g_es),
                    )
                )
            create_database_structure(
                group, data_augmentation, registering, p_ed, g_ed, p_es, g_es, data_mid=p_mid, gt_mid=g_mid
            )


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
    data_processing_group = parser.add_mutually_exclusive_group()
    data_processing_group.add_argument(
        "-d", "--data_augmentation", action="store_true", help="Add data augmentation (rotation -60 to 60)."
    )
    data_processing_group.add_argument(
        "-r",
        "--registering",
        action="store_true",
        help="Apply registering (registering and rotation)." "Only works when groundtruths are provided.",
    )
    args = parser.parse_args()
    generate_dataset(args.path, args.name, args.data_augmentation, args.registering)


if __name__ == "__main__":
    main()
