# -*- coding: utf-8 -*-

"""
This file contains any helpful generic functions concerning datasets.
"""

from collections import Counter

import numpy as np
from skimage.util import crop
from sklearn.feature_extraction.image import extract_patches_2d


def extract_balanced_sets(labels, sets_proportion, seed=1234):
    """ Splits a dataset into n different balanced subsets with the same number of subjects per class.

    Args:
        labels: list or ndarray of discrete values, Labels of every subjects (of size n = number of subjects).
        sets_proportion: list or tuple of int, Percent of subjects to be contained in each set.
        seed: int, Seed for the random number generator (optional).

    Returns:
        A ndarray of size n with a non-zero index value of the set assigned for
        each subject.
    """

    num_subjects = len(labels)
    subject_ids = np.arange(num_subjects)
    set_assignment = np.zeros(num_subjects, dtype=int)
    rng = np.random.RandomState(seed)

    # First, find label with fewest number of subjects.
    labels_counter = Counter(labels)
    different_labels = sorted(labels_counter.keys())
    min_subjects_in_label = min(labels_counter.values())

    # Then, extract that number of subjects for each label + set.
    for label_id, label in enumerate(different_labels):
        subjects_ids_in_label = subject_ids[labels == label]
        rng.shuffle(subjects_ids_in_label)
        prev_index = 0

        for set_id, set_proportion in enumerate(sets_proportion):
            num_subjects_to_keep = set_proportion / 100. * min_subjects_in_label
            next_index = prev_index + int(np.floor(num_subjects_to_keep))

            subjects_to_keep = subjects_ids_in_label[prev_index:next_index]
            set_assignment[subjects_to_keep] = set_id + 1

            prev_index = next_index

    return set_assignment


def standardize_data(data, axis=0):
    """ Performs standardization independently on every axis of the data.

    Args
        data: ndarray, Whole dataset.
        axis: int, Subjects' axis.

    Returns:
        ndarray of the same shape as input data.
    """

    mean = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis)

    # Set standard deviations of zero to 1 to avoid division by zero.
    std[std == 0] = 1.

    return (data - mean) / std


def vectorize_data(data, axis=0):
    """ Turns multi-dimensional data into a 1D vector per subject.

    Args:
        data: ndarray, Whole dataset.
        axis: int, Subjects' axis.

    Returns:
        2D ndarray of shape num_subjects x num_features
    """

    # Put subject's axis first.
    if axis != 0:
        data = np.swapaxes(data, 0, axis)

    feature_dims = data.shape[1:]
    num_features = np.product(feature_dims)
    num_subjects = data.shape[0]

    return data.reshape(num_subjects, num_features)


def get_patch_at_x_y(data, patch, x, y):
    """ Get a 2D patch from the data

    Args:
        data: ndarray, The whole image.
        patch: list or tuple, Size of the patches to extract.
        x: int, x coordinate where to extract patches.
        y: int, y coordinate where to extract patches.

    Returns:
        patches at x, y
    """
    patches, output_shape = get_patch_list(data, patch, stride=[1, 1, 1])
    # Scikit-learn adds an additional unneeded dimension so remove it
    patches = np.squeeze(patches, axis=len(patches.shape) - 3)
    return patches[:, x, y]


def get_patch_list(data, patch, stride):
    """ Get all the patches of the given size and stride.

    Args:
        data: ndarray, Data to extract patches.
        patch: list or tuple, Size of patches to extract.
        stride: list or tuple, Stride during patch extraction.

    Returns:
        tuple with two elements:
            - 1st is an array of patches
            - 2nd is the shape of the input image
    """

    # Add channel dimension if missing
    if len(patch) != data.ndim:
        patch = (1,) + tuple(patch)
    if len(stride) != data.ndim:
        stride = (1,) + tuple(stride)

    res = extract_patches_2d(data, patch, stride)
    return res, data.shape


def centered_padding(image, pad_size, c_val=0):
    """ Pad the image given in parameters to have a size of self.image_size.

    Args:
        image: ndarray (3d or 4d), Numpy array of data to be padded.
        pad_size: list or tuple, Size of the image after padding.
        c_val: int or float, Value used for padding.

    Returns:
        A ndarray (3D or 4D) padded with a size of pad_size.
    """
    im_size = np.array(pad_size)

    if image.ndim == 4:
        to_pad = (im_size - image.shape[1:3]) // 2
        to_pad = np.array(to_pad).astype(np.int)
        to_pad = ((0, 0), (to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))
    else:
        to_pad = (im_size - image.shape[:2]) // 2
        to_pad = np.array(to_pad).astype(np.int)
        to_pad = ((to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))

    return np.pad(image, to_pad, mode='constant', constant_values=c_val)


def centered_crop(image, crop_size):
    """ Crop the image given in parameters to have a size of crop_size.

    Args:
        image: ndarray (4D), Numpy array of data to be padded.
        crop_size: list or tuple, Define the new dimension of the image.

    Returns:
        A ndarray (4D) cropped of the size of crop_size.
    """

    if image.ndim == 4:
        to_crop = (np.array(image.shape[1:3]) - crop_size) // 2
        to_crop = np.array(to_crop, dtype=np.int)
        to_crop = ((0, 0), (to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    else:
        to_crop = (np.array(image.shape[:2]) - crop_size) // 2
        to_crop = np.array(to_crop, dtype=np.int)
        to_crop = ((to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    return crop(image, to_crop)


def centered_resize(image, size, c_val=0):
    """ Centered image resize using crop or padding with c_val.

    Args:
        image: ndarray, A 3d or 4d numpy array of the image.
        size: iterable of int, The output size of the input image.
        c_val: int or float, The value used for the padding.

    Returns:
        A numpy array with the needed output size.
    """

    if image.ndim == 4:
        isize = image.shape[1:3]
    else:
        isize = image.shape[:2]

    # Check the first dimension to select if we crop of pad
    if size[0] - isize[0] < 0:
        image = centered_crop(image, [size[0], isize[1]])
    elif size[0] - isize[0] > 0:
        image = centered_padding(image, [size[0], isize[1]], c_val)

    # Check if we crop or pad along the second dim of the image
    if size[1] - isize[1] < 0:
        image = centered_crop(image, size)
    elif size[1] - isize[1] > 0:
        image = centered_padding(image, size, c_val)

    return image


def preprocess_channel(img, preproc_fn):
    """This method preprocesses an image by channel.

    Args:
        img: hdf5 data, img in 3d like this ('s', 0, 1, 'c'):
                - 's': number of slices.
                - 0, 1: x and y dim.
                - 'c': number of channels.
        preproc_fn: list of functions used for preprocessing

    Returns:
        ndarray of the image with preprocesses applied on each channel.
    """

    # Convert the image in float32 (GPU limitation)
    img = np.array(img).astype(np.float32)

    # Preprocessing over each channel
    for c in range(img.shape[-1]):
        for pre in preproc_fn:
            img[..., c] = pre(img[..., c])

    return img


def bbox(segmentation, labels, bbox_margin=0.05):
    """ Computes the coordinates of a bounding box (bbox) around a region of interest (ROI).

    Args:
        segmentation: ndarray, segmentation in which to identify the coordinates of the bbox.
        labels: int or list, labels of the classes that are part of the ROI.
        bbox_margin: float, ratio by which to enlarge the bbox from the closest possible fit, so as to leave a
                     slight margin at the edges of the bbox.

    Returns:
        ndarray, coordinates of the bbox, in the following order: row_min, col_min, row_max, col_max.
    """
    # Only keep ROI from the groundtruth
    roi_mask = np.isin(np.argmax(segmentation, axis=-1), labels)

    # Find the coordinates of the bounding box around the ROI
    rows = np.any(roi_mask, axis=1)
    cols = np.any(roi_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Compute the size of the margin between the ROI and its bounding box
    dy = int(bbox_margin * (col_max - col_min))
    dx = int(bbox_margin * (row_max - row_min))

    # Apply margin to bbox coordinates
    row_min, row_max = row_min - dx, row_max + dx
    col_min, col_max = col_min - dy, col_max + dy

    # Check limits
    row_min, row_max = np.max([0, row_min]), np.min([row_max, segmentation.shape[0] - 1])
    col_min, col_max = np.max([0, col_min]), np.min([col_max, segmentation.shape[1] - 1])

    return np.array([row_min, col_min, row_max, col_max])
