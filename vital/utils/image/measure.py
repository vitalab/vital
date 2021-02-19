from numbers import Real
from typing import Tuple, TypeVar

import numpy as np
from torch import Tensor

from vital.data.config import SemanticStructureId
from vital.utils.decorators import auto_cast_data

T = TypeVar("T", np.ndarray, Tensor)


class Measure:
    """Generic implementations of various measures on images represented as numpy arrays or torch tensors."""

    @staticmethod
    @auto_cast_data
    def structure_area(segmentation: T, labels: SemanticStructureId) -> T:
        """Computes the number of pixels, in a segmentation map, associated to a structure.

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the number of pixels of the structure.
            labels: Labels of the classes that are part of the structure for which to count the number of pixels.

        Returns:
            ([N], 1), Number of pixels associated to the structure, in each segmentation of the batch.
        """
        return np.isin(segmentation, labels).sum((-2, -1))[..., None]

    @staticmethod
    @auto_cast_data
    def bbox(segmentation: T, labels: SemanticStructureId, bbox_margin: Real = 0.05, normalize: bool = False) -> T:
        """Computes the coordinates of a bounding box (bbox) around a region of interest (ROI).

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the coordinates of the bbox.
            labels: Labels of the classes that are part of the ROI.
            bbox_margin: Ratio by which to enlarge the bbox from the closest possible fit, so as to leave a slight
                margin at the edges of the bbox.
            normalize: If ``True``, normalizes the bbox coordinates from between 0 and H or W to between 0 and 1.

        Returns:
            ([N], 4), Coordinates of the bbox, in the following order: row_min, col_min, row_max, col_max.
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

    @staticmethod
    @auto_cast_data
    def denormalize_bbox(roi_bbox: T, output_size: Tuple[int, int], check_bounds: bool = False) -> T:
        """Gives the pixel-indices of a bounding box (bbox) w.r.t an output size based on the bbox's normalized coord.

        Args:
            roi_bbox: ([N], 4), Normalized coordinates of the bbox, in the following order:
                row_min, col_min, row_max, col_max.
            output_size: (H, W), Size for which to compute pixel-indices based on the normalized coordinates.
            check_bounds: If ``True``, perform various checks on the denormalized coordinates:
                - ensure they fit between 0 and H or W
                - ensure that the min bounds are smaller than the max bounds
                - ensure that the bbox is at least one pixel wide in each dimension

        Returns:
            ([N], 4), Coordinates of the bbox, in the following order: row_min, col_min, row_max, col_max.
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
