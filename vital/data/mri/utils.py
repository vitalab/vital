"""This file contains any helpful generic functions concerning MRI datasets."""
from typing import Tuple

import numpy as np
from skimage.util import crop


def to_onehot(matrix: np.ndarray, nb_classes: int):
    """Transform a matrix containing integer label class into a matrix containing one hot class labels.

    The last dim of the matrix should be the category (classes).

    Args:
        matrix: A numpy matrix to convert into a categorical matrix.
        nb_classes: int, number of classes

    Returns:
        A numpy array representing the categorical matrix of the input.
    """
    return matrix == np.arange(nb_classes)[np.newaxis, np.newaxis, np.newaxis, :]


def centered_padding(image: np.ndarray, pad_size: Tuple[int, int], c_val: float = 0) -> np.ndarray:
    """Pad the image given in parameters to have a size of self.image_size.

    Args:
        image: Numpy array (3d or 4d) of data to be padded.
        pad_size: Size of the image after padding.
        c_val: Value used for padding.

    Returns:
        3D or 4D image padded with a size of pad_size.
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

    return np.pad(image, to_pad, mode="constant", constant_values=c_val)


def centered_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """Crop the image given in parameters to have a size of crop_size.

    Args:
        image: 4D Numpy array of data to be padded.
        crop_size: Defines the new dimension of the image.

    Returns:
        4D image cropped of the size of crop_size.
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


def centered_resize(image: np.ndarray, size: Tuple[int, int], c_val: float = 0) -> np.ndarray:
    """Centered image resize using crop or padding with c_val.

    Args:
        image: A 3d or 4d numpy array of the image.
        size: The output size of the input image.
        c_val: The value used for the padding.

    Returns:
        Image with the needed output size.
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


def centered_resize_gt(gt: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Center and resize Gt.

    Args:
        gt: gt one hot segmentation map
        size: size of the segmentation map after resize.

    Returns:
        Centered and resized gt.
    """
    gt = centered_resize(gt, size)

    # Need to redo the background class due to resize
    # that set the image border to 0
    summed = np.clip(gt[..., 1:].sum(axis=-1), 0, 1)
    gt[..., 0] = np.abs(1 - summed)

    return gt
