from typing import Sequence, Tuple

import numpy as np
import PIL
import torch
from PIL import Image
from PIL.Image import NEAREST
from torch import Tensor


def resize_image(image: np.ndarray, size: Tuple[int, int], resample: PIL.Image = NEAREST) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: input image to process. Must be in a format supported by PIL.
        size: width and height dimensions of the processed image to output.
        resample: resampling filter to use.

    Returns:
        input image resized to the specified dimensions.
    """
    resized_image = np.array(Image.fromarray(image).resize(size, resample=resample))
    return resized_image


def remove_labels(segmentation: np.ndarray, labels_to_remove: Sequence[int], fill_label: int = 0) -> np.ndarray:
    """Removes labels from the segmentation map, reassigning the affected pixels to `fill_label`.

    Args:
        segmentation: ([N], H, W, [1|C]), segmentation map from which to remove labels.
        labels_to_remove: labels to remove.
        fill_label: label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ([N], H, W, 1), categorical segmentation map with the specified labels removed.
    """
    seg = segmentation.copy()
    if seg.max() == 1 and seg.shape[-1] > 1:  # If the segmentation map is in one-hot format
        for label_to_remove in labels_to_remove:
            seg[..., fill_label] += seg[..., label_to_remove]
        seg = np.delete(seg, labels_to_remove, axis=-1)
    else:  # the segmentation map is categorical
        seg[np.isin(seg, labels_to_remove)] = fill_label
    return seg


def segmentation_to_tensor(segmentation: np.ndarray, dtype: str = "float32") -> Tensor:
    """Converts a segmentation map to a tensor, including reordering the dimensions.

    Args:
        segmentation: (H, W, C), segmentation map to convert to a tensor.
        dtype: The data type expected for the converted tensor, as a string
               (`float32`, `float64`, `int32`...)

    Returns:
        (C, H, W), segmentation map converted to a tensor.
    """
    return torch.from_numpy(segmentation.transpose((2, 0, 1)).astype(dtype))
