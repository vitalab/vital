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
        image: Input image to process. Must be in a format supported by PIL.
        size: Width and height dimensions of the processed image to output.
        resample: Resampling filter to use.

    Returns:
        Input image resized to the specified dimensions.
    """
    resized_image = np.array(Image.fromarray(image).resize(size, resample=resample))
    return resized_image


def remove_labels(segmentation: np.ndarray, labels_to_remove: Sequence[int], fill_label: int = 0) -> np.ndarray:
    """Removes labels from the segmentation map, reassigning the affected pixels to `fill_label`.

    Args:
        segmentation: ([N], H, W, [1|C]), Segmentation map from which to remove labels.
        labels_to_remove: Labels to remove.
        fill_label: Label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ([N], H, W, [1]), Categorical segmentation map with the specified labels removed.
    """
    seg = segmentation.copy()
    if seg.max() == 1 and seg.shape[-1] > 1:  # If the segmentation map is in one-hot format
        for label_to_remove in labels_to_remove:
            seg[..., fill_label] += seg[..., label_to_remove]
        seg = np.delete(seg, labels_to_remove, axis=-1)
    else:  # the segmentation map is categorical
        seg[np.isin(seg, labels_to_remove)] = fill_label
    return seg


def segmentation_to_tensor(segmentation: np.ndarray, dtype: str = "int64") -> Tensor:
    """Converts a segmentation map to a tensor, including reordering the dimensions.

    Args:
        segmentation: (H, W, [C]), Segmentation map to convert to a tensor.
        dtype: Data type expected for the converted tensor, as a string
            (`float32`, `float64`, `int32`...).

    Returns:
        ([C], H, W), Segmentation map converted to a tensor.
    """
    if len(segmentation.shape) == 3:  # If there is a specific channel dimension
        # Change format from channel last (H, W, C) to channel first (C, H, W)
        segmentation = segmentation.transpose((2, 0, 1))
    return torch.from_numpy(segmentation.astype(dtype))
