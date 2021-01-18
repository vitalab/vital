from functools import wraps
from numbers import Real
from typing import Callable, Tuple, Type, TypeVar

import numpy as np
import torch
from skimage import measure
from skimage.measure._regionprops import RegionProperties
from torch import Tensor

from vital.data.config import SemanticStructureId

T = TypeVar("T", np.ndarray, Tensor)
DATA_TYPES = [np.ndarray, Tensor]


def _check_data_type(data: T) -> Type[T]:
    """Inspects the type of `data` and ensures it is part of `DATA_TYPES`.

    Args:
        data: Data whose type to inspect.

    Returns:
        The type of data, if it is part of `DATA_TYPES`.

    Raises:
        ValueError: If the type of data is not part of `DATA_TYPES`.
    """
    dtype = type(data)
    if dtype not in DATA_TYPES:
        raise ValueError(
            f"The `Measure` API is not supported for data of type '{dtype}'. Either provide the implementation "
            f"of the API for your target data type, or cast your data to one of the following supported types: "
            f"{DATA_TYPES}."
        )
    return dtype


def auto_cast_data(cls_method: Callable) -> Callable:
    """Decorator to allow `Measure` classmethods relying on numpy arrays to accept any `DATA_TYPES`.

    Args:
        cls_method: `Measure` classmethod to wrap.

    Returns:
        Classmethod that can accept any `DATA_TYPES` by converting between them and numpy arrays.
    """

    @wraps(cls_method)
    def _call_func_with_cast_data(cls, data, *args, **kwargs):
        dtype = _check_data_type(data)
        if dtype == Tensor:
            data_device = data.device
            data = data.detach().cpu().numpy()
        result = cls_method(cls, data, *args, **kwargs)
        if dtype == Tensor:
            result = torch.tensor(result, device=data_device)
        return result

    return _call_func_with_cast_data


class Measure:
    """Generic implementations of various measures on images represented as numpy arrays or torch tensors."""

    @classmethod
    def _region_prop(
        cls, segmentation: np.ndarray, labels: SemanticStructureId, prop_fn: Callable[[RegionProperties], Real]
    ) -> np.ndarray:
        """Abstract method to compute a property of a structure in a (batch of) segmentation map(s).

        Args:
            segmentation: ([N], H, W), Segmentation map(s).
            labels: Labels of the classes that are part of the structure of interest.
            prop_fn: Function that computes the desired property from the segmentation's `RegionProperties`.

        Returns:
            ([N], 1), Property of the structure, in each segmentation of the batch.
        """
        if is_single_sample := segmentation.ndim == 2:  # If we don't have a batch of segmentations
            segmentation = segmentation[None]
        prop = np.array(
            [
                prop_fn(measure.regionprops(binary_sample)[0])
                for binary_sample in np.isin(segmentation, labels).astype(np.uint8)
            ]
        )
        if not is_single_sample:
            prop = prop[..., None]
        return prop

    @classmethod
    @auto_cast_data
    def bbox(cls, segmentation: T, labels: SemanticStructureId, bbox_margin: Real = 0.05, normalize: bool = False) -> T:
        """Computes the coordinates of a bounding box (bbox) around a region of interest (ROI).

        Args:
            segmentation: (H, W), Segmentation in which to identify the coordinates of the bbox.
            labels: Labels of the classes that are part of the ROI.
            bbox_margin: Ratio by which to enlarge the bbox from the closest possible fit, so as to leave a slight
                margin at the edges of the bbox.
            normalize: If ``True``, normalizes the bbox coordinates from between 0 and H or W to between 0 and 1.

        Returns:
            Coordinates of the bbox, in the following order: row_min, col_min, row_max, col_max.
        """
        # Only keep ROI from the groundtruth
        roi_mask = np.isin(segmentation, labels)

        # Find the coordinates of the bounding box around the ROI
        rows = roi_mask.any(1)
        cols = roi_mask.any(0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Compute the size of the margin between the ROI and its bounding box
        dy = int(bbox_margin * (col_max - col_min))
        dx = int(bbox_margin * (row_max - row_min))

        # Apply margin to bbox coordinates
        row_min, row_max = row_min - dx, row_max + dx + 1
        col_min, col_max = col_min - dy, col_max + dy + 1

        # Check limits
        row_min, row_max = max(0, row_min), min(row_max, roi_mask.shape[0])
        col_min, col_max = max(0, col_min), min(col_max, roi_mask.shape[1])

        roi_bbox = np.array([row_min, col_min, row_max, col_max])

        if normalize:
            roi_bbox = roi_bbox.astype(float)
            roi_bbox[[0, 2]] = roi_bbox[[0, 2]] / segmentation.shape[0]  # Normalize height
            roi_bbox[[1, 3]] = roi_bbox[[1, 3]] / segmentation.shape[1]  # Normalize width

        return roi_bbox

    @classmethod
    @auto_cast_data
    def denormalize_bbox(cls, roi_bbox: T, output_size: Tuple[int, int], check_bounds: bool = False) -> T:
        """Gives the pixel-indices of a bounding box (bbox) w.r.t an output size based on the bbox's normalized coord.

        Args:
            roi_bbox: (N, 4), Normalized coordinates of the bbox, in the following order:
                row_min, col_min, row_max, col_max.
            output_size: (H, W), Size for which to compute pixel-indices based on the normalized coordinates.
            check_bounds: If ``True``, perform various checks on the denormalized coordinates:
                - ensure they fit between 0 and H or W
                - ensure that the min bounds are smaller than the max bounds
                - ensure that the bbox is at least one pixel wide in each dimension

        Returns:
            Coordinates of the bbox, in the following order: row_min, col_min, row_max, col_max.
        """
        # Copy input data to ensure we don't write over user data
        roi_bbox = np.copy(roi_bbox)

        if check_bounds:
            # Clamp predicted RoI bbox to ensure it won't end up out of range of the image
            roi_bbox = np.clip(roi_bbox, 0, 1)

        # Change ROI bbox from normalized between 0 and 1 to absolute pixel coordinates
        roi_bbox[:, (0, 2)] = (roi_bbox[:, (0, 2)] * output_size[0]).round()  # Height
        roi_bbox[:, (1, 3)] = (roi_bbox[:, (1, 3)] * output_size[1]).round()  # Width

        if check_bounds:
            # Clamp predicted min bounds are at least two pixels smaller than image bounds
            # to allow for inclusive upper bounds
            roi_bbox[:, 0] = np.minimum(roi_bbox[:, 0], output_size[0] - 1)  # Height
            roi_bbox[:, 1] = np.minimum(roi_bbox[:, 1], output_size[1] - 1)  # Width

            # Clamp predicted max bounds are at least one pixel bigger than min bounds
            roi_bbox[:, 2] = np.maximum(roi_bbox[:, 2], roi_bbox[:, 0] + 1)  # Height
            roi_bbox[:, 3] = np.maximum(roi_bbox[:, 3], roi_bbox[:, 1] + 1)  # Width

        return roi_bbox
