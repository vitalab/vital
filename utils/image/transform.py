from typing import Tuple, Sequence

import PIL
import numpy as np
from PIL import Image
from PIL.Image import NEAREST


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
        segmentation: ([N], H, W, 1/C), segmentation map from which to remove labels.
        labels_to_remove: labels to remove.
        fill_label: label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ([N], H, W, 1), labelled segmentation map with the specified labels removed.
    """
    segmentation = segmentation.copy()

    if segmentation.shape[-1] == 1:  # If the segmentation map is labelled
        segmentation[np.isin(segmentation, labels_to_remove)] = fill_label

    else:  # the segmentation map is in one-hot format
        for label_to_remove in labels_to_remove:
            segmentation[..., fill_label] += segmentation[..., label_to_remove]
        segmentation = np.delete(segmentation, labels_to_remove, axis=-1)

    return segmentation
