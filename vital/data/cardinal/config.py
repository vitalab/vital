from enum import auto, unique
from typing import Tuple

from strenum import SnakeCaseStrEnum, UppercaseStrEnum

from vital.data.config import LabelEnum

PATIENT_ID_REGEX = r"\d{4}"
HDF5_FILENAME_PATTERN = "{patient_id}_{view}.h5"
IMG_FILENAME_PATTERN = "{patient_id}_{view}_{tag}{ext}"
IMG_FORMAT = ".nii.gz"
IMG_ATTRS_FORMAT = "npz"
ATTRS_FILENAME_PATTERN = "{patient_id}{ext}"
ATTRS_FORMAT = "yaml"

IN_CHANNELS: int = 1
"""Number of input channels of the images in the dataset."""

DEFAULT_SIZE: Tuple[int, int] = (256, 256)
"""Default size at which the raw B-mode images are resized."""


@unique
class Label(LabelEnum):
    """Identifiers of the different anatomical structures available in the dataset's segmentation mask."""

    BG = 0
    """BackGround"""
    LV = 1
    """Left Ventricle"""
    MYO = 2
    """MYOcardium"""


@unique
class View(UppercaseStrEnum):
    """Names of the different views available for each patient."""

    A4C = auto()
    """Apical 4 Chamber"""
    A2C = auto()
    """Apical 2 Chamber"""
    A3C = auto()
    """Apical 3 Chamber"""


@unique
class ImageAttribute(SnakeCaseStrEnum):
    """Names of the cardiac shape attributes extracted from the image data."""

    gls = auto()
    """Global Longitudinal Strain (GLS) of the endocardium."""
    lv_area = auto()
    """Number of pixels covered by the left ventricle (LV)."""
    lv_length = auto()
    """Distance between the LV's base and apex."""
    myo_area = auto()
    """Number of pixels covered by the myocardium (MYO)."""


@unique
class ClinicalAttribute(SnakeCaseStrEnum):
    """Name of the patient's scalar clinical attributes extracted from either their record or the images."""

    ef = auto()
    """Ejection Fraction (EF)."""
    edv = auto()
    """End-Diastolic Volume (EDV)."""
    esv = auto()
    """End-Systolic Volume (ESV)."""


class CardinalTag(SnakeCaseStrEnum):
    """Tags referring to the different type of data stored."""

    # Tags describing data modalities
    image_attrs = auto()
    clinical_attrs = auto()

    # Tags referring to image data
    bmode = auto()
    mask = auto()
    resized_bmode = f"{bmode}_{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}"
    resized_mask = f"{mask}_{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}"

    # Tags for prefix/suffix to add for specific
    post = auto()

    # Tags referring to data attributes
    voxelspacing = auto()
