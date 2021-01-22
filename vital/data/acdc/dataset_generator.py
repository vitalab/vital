import argparse
import os
from glob import glob
from os.path import basename

import h5py
import nibabel as nib
import numpy as np
from natsort import natsorted
from scipy import ndimage
from scipy.ndimage.interpolation import rotate

from vital.data.acdc.config import AcdcTags, Instant, Label
from vital.data.acdc.utils.acdc import AcdcRegisteringTransformer
from vital.data.acdc.utils.utils import centered_resize
from vital.data.config import Subset

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


def generate_list_directory(path):
    """Generate the list of nifti images from the path.

    Args:
        path: string, path to nifti images
    """
    if not os.path.exists(path):
        return []

    path = os.path.join(path, "*", "*")
    paths = natsorted(glob(path))
    paths = [path for path in paths if "Info" not in path]
    paths = [path for path in paths if path.find("_4d") < 0]
    return paths


def _to_categorical(matrix, nb_classes):
    """Transform a matrix containing integer label class into a matrix containing categorical class labels.

    The last dim of the matrix should be the category (classes).

    Args:
        matrix: ndarray, A numpy matrix to convert into a categorical matrix.
        nb_classes: int, number of classes

    Returns:
        A numpy array representing the categorical matrix of the input.
    """
    return matrix == np.arange(nb_classes)[np.newaxis, np.newaxis, np.newaxis, :]


def _mass_center(imgs):
    """Function to extract the center of masses of a 3D ground truth image.

    Args:
        imgs: images
    """
    centers = np.array([ndimage.measurements.center_of_mass(img[:, :, 3]) for img in imgs])
    # Need to fix the Nan when the ground truth slice is a slice of zeros.
    # Set it to center 256 // 2 = 128
    centers[np.isnan(centers)] = 128
    return centers.astype(np.int16)


def _generate_centered_prob_map(image, shape, center, label):
    """Function to extract the information from the ground truth image given the centers.

    Args:
        image: np.array, Numpy array of the ground truth.
        shape: np.array, Shape of the desired prior image.
        center: np.array, Numpy array of shape (slices, 2).
        label: int, Which label to extract from the ground truth.

    Returns:
        Array of shape (slices, 100, 100, 1)
    """
    res = np.zeros(shape)
    # Nearest neighbour slice index between the number of slice
    # of the image and the ground truth
    space = np.linspace(0, shape[0] - 1, num=image.shape[0]).astype(np.int32)
    for i, (s, c) in enumerate(zip(space, center)):
        res[s] += image[
            i,
            c[0] - PRIOR_HALF_SIZE : c[0] + PRIOR_HALF_SIZE,
            c[1] - PRIOR_HALF_SIZE : c[1] + PRIOR_HALF_SIZE,
            label : label + 1,
        ]
    return res


def generate_probability_map(h5f, group):
    """Generate the probability map from all unrotated training exemples.

    Args:
    h5f: hdf5 File, Handle of the hdf5 file containing all the dataset.
    group: hdf5 File, Group where to create the prior shape (train, valid or test) train should be the
        default.
    """
    patient_keys = [key for key in group.keys() if key.endswith("_0")]
    image_keys = []
    for k1 in patient_keys:
        for k2 in group[k1].keys():
            image_keys.append("{}/{}/{}".format(k1, k2, AcdcTags.gt))

    images = [centered_resize(group[k][:], (256, 256)) for k in image_keys]
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


# def create_instant(patient_group, instant_name, img, gt, rot, registering_transformer):
#     instant = patient_group.create_group(Instant.ED.value)
#     # instant.attrs['voxel_size'] = ni_img.header.get_zooms()
#
#     r_img = rotate(img, rot, axes=(1, 2), reshape=False)
#     r_img = np.clip(r_img, img.min(), img.max())
#     r_img[np.isclose(r_img, 0.)] = 0.
#     if registering_transformer is not None:
#         registering_parameters, edg_img, r_img = registering_transformer.register_batch(img, r_img)
#         instant.attrs.update(registering_parameters)
#
#     instant.create_dataset(AcdcTags.img, data=r_img)
#
#     if gt:
#         r_img = rotate(img, rot, axes=(1, 2), output=np.uint8, reshape=False)
#         instant.create_dataset(AcdcTags.img, data=r_img)


def create_database_structure(
    group, data_augmentation, registering, data_ed, gt_ed, data_es, gt_es, data_mid=None, gt_mid=None
):
    """Create the dataset for the End-Systolic and End-Diastolic phases.

    If some data augmentation is involved we create also the rotation for each phase.

    Args:
        group: hdf5 Group, Group where we add each image by its name and the rotation associated to it.
        data_augmentation: bool, Enable/Disable data augmentation.
        registering: bool, Enable/Disable registering.
        data_ed: string, Path of the nifti diastolic MRI image.
        gt_ed: string, Path of the nifti diastolic MRI segmentation of data_ed.
        data_es: string, Path of the nifti systolic MRI image.
        gt_es: string, Path of the nifti systolic MRI segmentation of data_ed.
        data_mid: Path of the nifti MRI image at a time step between ED and ES.
        gt_mid: Path of the nifti MRI segmentation of data_mid.

    """
    p_name = data_ed.split(os.path.sep)[-2]

    ni_img = nib.load(data_ed)

    ed_img = ni_img.get_data().astype(np.float32)
    ed_img = ed_img.transpose(2, 0, 1)[..., np.newaxis]

    if gt_ed:
        ni_img = nib.load(gt_ed)
        edg_img = ni_img.get_data()
        edg_img = edg_img.transpose(2, 0, 1)[..., np.newaxis]
        edg_img = _to_categorical(edg_img, Label.count()).astype(np.uint8)

    ni_img = nib.load(data_es)
    es_img = ni_img.get_data().astype(np.float32)
    es_img = es_img.transpose(2, 0, 1)[..., np.newaxis]

    if gt_es:
        ni_img = nib.load(gt_es)
        esg_img = ni_img.get_data()
        esg_img = esg_img.transpose(2, 0, 1)[..., np.newaxis]
        esg_img = _to_categorical(esg_img, Label.count()).astype(np.uint8)

    if data_mid:
        ni_img = nib.load(data_mid)
        mid_img = ni_img.get_data().astype(np.float32)
        mid_img = mid_img.transpose(2, 0, 1)[..., np.newaxis]
        if gt_mid:
            ni_img = nib.load(gt_mid)
            midg_img = ni_img.get_data()
            midg_img = midg_img.transpose(2, 0, 1)[..., np.newaxis]
            midg_img = _to_categorical(midg_img, Label.count()).astype(np.uint8)

    if data_augmentation:
        iterable = ROTATIONS
    else:
        iterable = [
            0,
        ]

    if registering:
        registering_transformer = AcdcRegisteringTransformer()

    for rot in iterable:
        patient = group.create_group("{}_{}".format(p_name, rot))
        # patient.attrs[AcdcTags.voxel_spacing] = ni_img.header.get_zooms()

        instant = patient.create_group(Instant.ED.value)
        instant.attrs["voxel_size"] = ni_img.header.get_zooms()

        # ED gate with gt
        r_img = rotate(ed_img, rot, axes=(1, 2), reshape=False)
        r_img = np.clip(r_img, ed_img.min(), ed_img.max())
        r_img[np.isclose(r_img, 0.0)] = 0.0
        if registering:
            registering_parameters, edg_img, r_img = registering_transformer.register_batch(edg_img, r_img)
            instant.attrs.update(registering_parameters)

        instant.create_dataset(AcdcTags.img, data=r_img)

        if gt_ed:
            r_img = rotate(edg_img, rot, axes=(1, 2), output=np.uint8, reshape=False)
            instant.create_dataset(AcdcTags.gt, data=r_img)

        instant = patient.create_group(Instant.ES.value)

        # ES gate with gt
        r_img = rotate(es_img, rot, axes=(1, 2), reshape=False)
        r_img = np.clip(r_img, es_img.min(), es_img.max())
        r_img[np.isclose(r_img, 0.0)] = 0.0

        if registering:
            registering_parameters, esg_img, r_img = registering_transformer.register_batch(esg_img, r_img)
            instant.attrs.update(registering_parameters)

        instant.create_dataset(AcdcTags.img, data=r_img)
        instant.attrs["voxel_size"] = ni_img.header.get_zooms()

        if gt_es:
            r_img = rotate(esg_img, rot, axes=(1, 2), output=np.uint8, reshape=False)
            instant.create_dataset(AcdcTags.gt, data=r_img)

        # Add mid-cycle data
        if data_mid:
            instant = patient.create_group(Instant.MID.value)
            instant.attrs["voxel_size"] = ni_img.header.get_zooms()

            # Mid gate with gt
            r_img = rotate(mid_img, rot, axes=(1, 2), reshape=False)
            r_img = np.clip(r_img, mid_img.min(), mid_img.max())
            r_img[np.isclose(r_img, 0.0)] = 0.0

            if registering:
                registering_parameters, midg_img, r_img = registering_transformer.register_batch(midg_img, r_img)
                instant.attrs.update(registering_parameters)

            instant.create_dataset(AcdcTags.img, data=r_img)

            if gt_mid:
                r_img = rotate(midg_img, rot, axes=(1, 2), output=np.uint8, reshape=False)
                instant.create_dataset(AcdcTags.gt, data=r_img)


def generate_dataset(path, name, data_augmentation=False, registering=False):
    """Function that generates each dataset, train, valid and test.

    Args:
        path: string, Path where we can find the images from the downloaded ACDC challenge.
        name: string, Name of the hdf5 file to generate.
        data_augmentation: bool, Enable/Disable data augmentation.
        registering: bool, Enable/Disable registering.

    # Raises:
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
    # indexes = np.arange(20)
    indexes = np.arange(5)

    train_idxs = []
    valid_idxs = []
    # 5 is the number of groups
    for i in range(5):
        start = i * 5
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

    print("train: ", len(train_paths))
    print("val: ", len(valid_paths))
    print("test: ", len(test_paths))

    with h5py.File(name, "w") as h5f:

        h5f.attrs[AcdcTags.registered] = registering

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
        "-d", "--data_augmentation", action="store_true", help="Add data augmentation (rotation)."
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
