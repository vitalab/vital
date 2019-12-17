import numpy as np
from PIL import Image
from PIL.Image import NEAREST


def resize_image(image, size, resample=NEAREST):
    """ Resizes the image to the specified dimensions.

    Args:
        image: ndarray, input image to process.
        size: tuple, width and height dimensions of the processed image to output.
        resample: resampling filter to use.

    Returns:
        ndarray, input image resized to the specified dimensions.
    """
    resized_image = np.array(Image.fromarray(image).resize(size, resample=resample))
    return resized_image


def resize_segmentation(segmentation, size, resample=NEAREST):
    """ Resizes the segmentation map to the specified dimensions.

    Args:
        segmentation: ndarray, segmentation map to process.
        size: tuple, width and height dimensions of the processed segmentation map to output.
        resample: resampling filter to use.

    Returns:
        ndarray, segmentation map resized to the specified dimensions.
    """
    # Ensure segmentation is in a format supported by Pillow (np.uint8)
    # to avoid possible "Cannot handle this data type" error
    resized_segmentation = np.array(Image.fromarray(segmentation.astype(np.uint8)).resize(size, resample=resample))
    # TOIMPROVE Apply mathematical opening to smooth the rough frontiers from the resize, if the resolution allows it
    return resized_segmentation


def onehot_remove_labels(segmentation, labels_to_remove, fill_label=0):
    """ Removes labels from the categorical segmentation map, reassigning the affected pixels to `fill_label`.

    Args:
        segmentation: ndarray, categorical segmentation map from which to remove labels.
        labels_to_remove: list, labels to remove.
        fill_label: int, label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ndarray, categorical segmentation map with the labels removed, meaning `len(labels_to_remove)` less channels
        in the channel dimension.
    """
    for label_to_remove in labels_to_remove:
        segmentation[..., fill_label] += segmentation[..., label_to_remove]
    segmentation = np.delete(segmentation, labels_to_remove, axis=-1)
    return segmentation


def remove_labels(segmentation, labels_to_remove, fill_label=0):
    """ Removes labels from the labelled segmentation map, reassigning the affected pixels to `fill_label`.

    Args:
        segmentation: ndarray, labelled segmentation map from which to remove labels.
        labels_to_remove: list, labels to remove.
        fill_label: int, label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ndarray, labelled segmentation map with the specified labels removed.
    """
    segmentation[np.isin(segmentation, labels_to_remove)] = fill_label
    return segmentation
