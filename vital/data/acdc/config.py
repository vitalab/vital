from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from vital.data.config import DataTag, Tags


class Label(DataTag):
    BG = 0
    RV = 1
    MYO = 2
    LV = 3


class Instant(DataTag):
    ED = 'ED'
    ES = 'ES'
    MID = 'MID'


@dataclass(frozen=True)
class AcdcTags(Tags):
    """ Class to gather the tags referring to the different types of data stored in the HDF5 datasets.

    Args:
        registered: name of the tag indicating whether the dataset was registered.
        voxel_spacing: name of the tag referring to metadata indicating the voxel size used in the output
        slice_index: name of the tag referring to the index of the slice
    """
    registered: str = 'register'

    voxel_spacing: str = 'voxel_size'
    slice_index: str = 'slice_index'

    proc_instants: str = "processed_instants"


image_size: int = 256
"""Dimension of the images in the dataset (in pixels)."""

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the ultrasound image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
