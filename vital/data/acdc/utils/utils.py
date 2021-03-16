"""This file contains any helpful generic functions concerning acdc dataset."""
from typing import Tuple

import numpy as np
from skimage.util import crop


def centered_pad(image: np.ndarray, pad_size: Tuple[int, int], pad_val: float = 0) -> np.ndarray:
    """Pads the image, or batch of images, so that (H, W) match the requested `pad_size`.

    Args:
        image: ([N], H, W, C), Data to be padded.
        pad_size: (H, W) of the image after padding.
        pad_val: Value used for padding.

    Returns:
        ([N], H, W, C), Image, or batch of images, padded so that (H, W) match `pad_size`.
    """
    im_size = np.array(pad_size)

    if image.ndim == 4:
        to_pad = (im_size - image.shape[1:3]) // 2
        to_pad = np.array(to_pad)
        to_pad = ((0, 0), (to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))
    else:
        to_pad = (im_size - image.shape[:2]) // 2
        to_pad = np.array(to_pad)
        to_pad = ((to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))

    return np.pad(image, to_pad, mode="constant", constant_values=pad_val)


def centered_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """Crops the image, or batch of images, so that (H, W) match the requested `crop_size`.

    Args:
        image: ([N], H, W, C), Data to be cropped.
        crop_size: (H, W) of the image after the crop.

    Returns:
         ([N], H, W, C), Image, or batch of images, cropped so that (H, W) match `crop_size`.
    """
    if image.ndim == 4:
        to_crop = (np.array(image.shape[1:3]) - crop_size) // 2
        to_crop = ((0, 0), (to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    else:
        to_crop = (np.array(image.shape[:2]) - crop_size) // 2
        to_crop = ((to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    return crop(image, to_crop)


def centered_resize(image: np.ndarray, size: Tuple[int, int], pad_val: float = 0) -> np.ndarray:
    """Centers image around the requested `size`, either cropping or padding to match the target size.

    Args:
        image:  ([N], H, W, C), Data to be adapted to fit the target (H, W).
        size: Target (H, W) for the input image.
        pad_val: The value used for the padding.

    Returns:
        ([N], H, W, C), Image, or batch of images, adapted so that (H, W) match `size`.
    """
    if image.ndim == 4:
        height, width = image.shape[1:3]
    else:
        height, width = image.shape[:2]

    # Check the height to select if we crop of pad
    if size[0] - height < 0:
        image = centered_crop(image, (size[0], width))
    elif size[0] - height > 0:
        image = centered_pad(image, (size[0], width), pad_val)

    # Check if we crop or pad along the width dim of the image
    if size[1] - width < 0:
        image = centered_crop(image, size)
    elif size[1] - width > 0:
        image = centered_pad(image, size, pad_val)

    return image
