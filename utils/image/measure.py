from typing import Union, List

import numpy as np


def bbox(segmentation,
         labels: Union[int, List[int]],
         bbox_margin: float = 0.05,
         channel_axis: int = -1) -> np.ndarray:
    """ Computes the coordinates of a bounding box (bbox) around a region of interest (ROI).

    Args:
        segmentation: arraylike, segmentation in which to identify the coordinates of the bbox.
        labels: labels of the classes that are part of the ROI.
        bbox_margin: ratio by which to enlarge the bbox from the closest possible fit, so as to leave a slight margin
                     at the edges of the bbox.
        channel_axis: axis of the channel dimension.

    Returns:
        coordinates of the bbox, in the following order: row_min, col_min, row_max, col_max.
    """
    # Only keep ROI from the groundtruth
    roi_mask = np.isin(np.argmax(segmentation, axis=channel_axis), labels)

    # Find the coordinates of the bounding box around the ROI
    rows = np.any(roi_mask, axis=1)
    cols = np.any(roi_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Compute the size of the margin between the ROI and its bounding box
    dy = int(bbox_margin * (col_max - col_min))
    dx = int(bbox_margin * (row_max - row_min))

    # Apply margin to bbox coordinates
    row_min, row_max = row_min - dx, row_max + dx + 1
    col_min, col_max = col_min - dy, col_max + dy + 1

    # Check limits
    row_min, row_max = max(0, row_min), min(row_max, segmentation.shape[0])
    col_min, col_max = max(0, col_min), min(col_max, segmentation.shape[1])

    return np.array([row_min, col_min, row_max, col_max])
