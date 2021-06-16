from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from vital.data.config import DataTag, Tags


class Label(DataTag):
    """Enumeration of tags related to the different anatomical structures segmented in the SLiver dataset.

    Attributes:
        BG: Label of the background.
        FG: Label of the liver.
    """

    BG = 0
    FG = 1


@dataclass(frozen=True)
class SLiverTags(Tags):
    """Class to gather the tags referring to the different types of data stored in the HDF5 datasets.

    Args:
        voxel_spacing: name of the tag referring to metadata indicating the voxel size used in the output
    """
    voxel_spacing: str = "voxel_size"



image_size: int = 256
"""Dimension of the images in the dataset (in pixels)."""

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the CT image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
