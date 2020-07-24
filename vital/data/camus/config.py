from typing import Any, Dict

import numpy as np

from vital.data.config import DataTag, Tags
from vital.utils.parameters import parameters


class Label(DataTag):
    """Enumeration of tags related to the different anatomical structures segmented in the dataset.

    Attributes:
        BG: Label of the background.
        ENDO: Label of the endocardium, delimiting the left ventricle (LV).
        EPI: Label of the epicardium, delimiting the myocardium (MYO).
        ATRIUM: Label of the left atrium.
    """

    BG = 0
    ENDO = 1
    EPI = 2
    ATRIUM = 3


class View(DataTag):
    """Enumeration of tags related to the different views available for each patient.

    Attributes:
        TWO: Tag referring to the two-chamber view.
        FOUR: Tag referring to the four-chamber view.
    """

    TWO = "2CH"
    FOUR = "4CH"


class Instant(DataTag):
    """Enumeration of tags related to noteworthy instants in the ultrasound sequences.

    Attributes:
        ED: Tag referring to the end-diastolic instant.
        ES: Tag referring to the end-systolic instant.
    """

    ED = "ED"
    ES = "ES"


@parameters
class CamusTags(Tags):
    """Class to gather the tags referring to CAMUS specific data, from both the training and result datasets.

    Args:
        registered: Tag indicating whether the dataset was registered.
        full_sequence: Tag indicating whether the dataset contains complete sequence between ED and ES for each view.
        img_proc: Tag referring to resized images, used as input when training models.
        gt_proc: Tag referring to resized groundtruths used as reference when training models.
        info: Tag referring to images' metadata.
        proc_instants: Tag referring to metadata indicating which image where affected by the postprocessing.
    """

    registered: str = "register"
    full_sequence: str = "sequence"

    img_proc: str = "img_proc"
    gt_proc: str = "gt_proc"
    info: str = "info"
    proc_instants: str = "processed_instants"


image_size: int = 256
"""Dimension of the images in the dataset (in pixels)."""

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the ultrasound image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
