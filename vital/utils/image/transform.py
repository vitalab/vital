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


def segmentation_to_tensor(segmentation: np.ndarray, flip_channels: bool = False, dtype: str = "int64") -> Tensor:
    """Converts a segmentation map to a tensor, including reordering the dimensions.

    Args:
        segmentation: ([N], H, W, [C]), Segmentation map to convert to a tensor.
        flip_channels: If ``True``, assumes that the input is in `channels_last` mode and will automatically convert it
            to `channels_first` mode. If ``False``, leaves the ordering of dimensions untouched.
        dtype: Data type expected for the converted tensor, as a string
            (`float32`, `float64`, `int32`...).

    Returns:
        ([N], [C], H, W), Segmentation map converted to a tensor.

    Raises:
        ValueError: When reordering from `channels_last` to `channel_first`, the segmentation provided is neither 2D nor
            3D (only shapes supported when reordering channels).
    """
    if flip_channels:  # If there is a specific channel dimension
        if len(segmentation.shape) == 3:  # If it is a single segmentation
            dim_to_transpose = (2, 0, 1)
        elif len(segmentation.shape) == 4:  # If there is a batch dimension to keep first
            dim_to_transpose = (0, 3, 1, 2)
        else:
            raise ValueError(
                "Segmentation to convert to tensor is expected to be a single segmentation (2D), "
                "or a batch of segmentations (3D): \n"
                f"The segmentation to convert is {len(segmentation.shape)}D."
            )
        # Change format from `channel_last`, i.e. ([N], H, W, C), to `channel_first`, i.e. ([N], C, H, W)
        segmentation = segmentation.transpose(dim_to_transpose)
    return torch.from_numpy(segmentation.astype(dtype))
