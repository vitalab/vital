from numbers import Real
from typing import Callable, Generic, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

from vital.data.config import SemanticStructureId

T = TypeVar("T", np.ndarray, Tensor)


class _Measure(Generic[T]):
    """Generic implementations of various measures on images represented as numpy arrays or torch tensors."""

    _backend = None
    _t_init_fn: Callable[..., T]
    _copy_fn: Callable[[T], T]
    _float_cast_fn: Callable[[T], T]
    _isin_fn: Callable[[T, SemanticStructureId], T]
    _clip_fn: Callable[[T, Real, Real], T]
    _elementwise_min: Callable[[T, Union[Real, T]], T]
    _elementwise_max: Callable[[T, Union[Real, T]], T]

    @classmethod
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
        roi_mask = cls._isin_fn(segmentation, labels)

        # Find the coordinates of the bounding box around the ROI
        rows = roi_mask.any(1)
        cols = roi_mask.any(0)
        row_min, row_max = cls._backend.where(rows)[0][[0, -1]]
        col_min, col_max = cls._backend.where(cols)[0][[0, -1]]

        # Compute the size of the margin between the ROI and its bounding box
        dy = int(bbox_margin * (col_max - col_min))
        dx = int(bbox_margin * (row_max - row_min))

        # Apply margin to bbox coordinates
        row_min, row_max = row_min - dx, row_max + dx + 1
        col_min, col_max = col_min - dy, col_max + dy + 1

        # Check limits
        row_min, row_max = max(0, row_min), min(row_max, roi_mask.shape[0])
        col_min, col_max = max(0, col_min), min(col_max, roi_mask.shape[1])

        roi_bbox = cls._t_init_fn([row_min, col_min, row_max, col_max])

        if normalize:
            roi_bbox = cls._float_cast_fn(roi_bbox)
            roi_bbox[[0, 2]] = roi_bbox[[0, 2]] / segmentation.shape[0]  # Normalize height
            roi_bbox[[1, 3]] = roi_bbox[[1, 3]] / segmentation.shape[1]  # Normalize width

        return roi_bbox

    @classmethod
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
        roi_bbox = cls._copy_fn(roi_bbox)

        if check_bounds:
            # Clamp predicted RoI bbox to ensure it won't end up out of range of the image
            roi_bbox = cls._clip_fn(roi_bbox, 0, 1)

        # Change ROI bbox from normalized between 0 and 1 to absolute pixel coordinates
        roi_bbox[:, (0, 2)] = (roi_bbox[:, (0, 2)] * output_size[0]).round()  # Height
        roi_bbox[:, (1, 3)] = (roi_bbox[:, (1, 3)] * output_size[1]).round()  # Width

        if check_bounds:
            # Clamp predicted min bounds are at least two pixels smaller than image bounds
            # to allow for inclusive upper bounds
            roi_bbox[:, 0] = cls._elementwise_min(roi_bbox[:, 0], output_size[0] - 1)  # Height
            roi_bbox[:, 1] = cls._elementwise_min(roi_bbox[:, 1], output_size[1] - 1)  # Width

            # Clamp predicted max bounds are at least one pixel bigger than min bounds
            roi_bbox[:, 2] = cls._elementwise_max(roi_bbox[:, 2], roi_bbox[:, 0] + 1)  # Height
            roi_bbox[:, 3] = cls._elementwise_max(roi_bbox[:, 3], roi_bbox[:, 1] + 1)  # Width

        return roi_bbox


class ArrayMeasure(_Measure[np.ndarray]):
    """Specialization of the generic ``_Measure`` methods to work with with numpy arrays."""

    _backend = np
    _t_init_fn = np.array
    _isin_fn = np.isin
    _clip_fn = np.clip
    _elementwise_min = np.minimum
    _elementwise_max = np.maximum

    @classmethod
    def _copy_fn(cls, array: np.ndarray) -> np.ndarray:
        return array.copy()

    @classmethod
    def _float_cast_fn(cls, array: np.ndarray) -> np.ndarray:
        return array.astype(float)


class TensorMeasure(_Measure[Tensor]):
    """Specialization of the generic ``_Measure`` methods to work with with torch tensors."""

    _backend = torch
    _t_init_fn = torch.tensor
    _clip_fn = torch.clamp

    @classmethod
    def _copy_fn(cls, tensor: Tensor) -> Tensor:
        return tensor.clone().detach()

    @classmethod
    def _float_cast_fn(cls, tensor: Tensor) -> Tensor:
        return tensor.float()

    @classmethod
    def _isin_fn(cls, tensor: Tensor, test_elements: SemanticStructureId) -> Tensor:
        return (tensor[..., None] == torch.tensor(test_elements, device=tensor.device)).any(-1)

    @classmethod
    def _elementwise_min(cls, tensor: Tensor, other: Union[Real, Tensor]) -> Tensor:
        return torch.min(tensor, torch.tensor(other).type_as(tensor))

    @classmethod
    def _elementwise_max(cls, tensor: Tensor, other: Union[Real, Tensor]) -> Tensor:
        return torch.max(tensor, torch.tensor(other).type_as(tensor))
