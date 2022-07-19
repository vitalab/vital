from dataclasses import dataclass
from enum import auto, unique
from typing import Any, Dict

import numpy as np
from strenum import LowercaseStrEnum

from vital.data.config import LabelEnum, Tags


@unique
class Label(LabelEnum):
    """Identifiers of the different anatomical structures available in the dataset's segmentation mask."""

    BG = 0
    """BackGround"""
    RV = 1
    """Right Ventricle"""
    MYO = 2
    """MYOcardium"""
    LV = 3
    """Left Ventricle"""


@unique
class Instant(LowercaseStrEnum):
    """Identifiers of noteworthy 3D volumes of 2D MRI slices."""

    ED = auto()
    """End-Diastolic volume"""
    ES = auto()
    """End-Systolic volume"""
    MID = auto()
    """Volume in the MIDdle between ED and ES"""


@dataclass(frozen=True)
class AcdcTags(Tags):
    """Class to gather the tags referring to the different types of data stored in the HDF5 datasets.

    Args:
        registered: name of the tag indicating whether the dataset was registered.
        voxel_spacing: name of the tag referring to metadata indicating the voxel size used in the output
        slice_index: name of the tag referring to the index of the slice
        proc_slices: Tag referring to metadata indicating which image were affected by the postprocessing.
    """

    registered: str = "register"
    voxel_spacing: str = "voxel_size"
    slice_index: str = "slice_index"
    proc_slices: str = "processed_slices"


image_size: int = 256
"""Dimension of the images in the dataset (in pixels)."""

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the MRI image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
