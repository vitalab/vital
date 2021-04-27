from dataclasses import dataclass
from typing import Any, Dict, Literal

import numpy as np

from vital.data.config import DataTag, Tags


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


@dataclass(frozen=True)
class View:
    """Collection of tags related to the different views available for each patient.

    Args:
        A2C: Tag referring to the apical two-chamber view.
        A4C: Tag referring to the apical four-chamber view.
    """

    A2C: str = "2CH"
    A4C: str = "4CH"


@dataclass(frozen=True)
class Instant:
    """Collection of tags related to noteworthy instants in ultrasound sequences.

    Args:
        ED: Tag referring to the end-diastolic instant.
        ES: Tag referring to the end-systolic instant.
    """

    @classmethod
    def from_sequence_type(cls, sequence_type: Literal["half_cycle", "full_cycle"]) -> "Instant":
        """Detects the specialized version of the `Instant` collection that fits the requested sequence type.

        Args:
            sequence_type: Flag that indicates the kind of sequences for which to provide the important instants.

        Returns:
            A specialized version of the `Instant` collection that fits the requested sequence type.
        """
        return globals()[f"{sequence_type.title().replace('_', '')}Instant"]()

    ED: str = "ED"
    ES: str = "ES"


@dataclass(frozen=True)
class HalfCycleInstant(Instant):
    """Collection of tags related to noteworthy instants in half-cycle ultrasound sequences."""

    pass


@dataclass(frozen=True)
class FullCycleInstant(Instant):
    """Collection of tags related to noteworthy instants in full-cycle ultrasound sequences.

    Args:
        ED_E: Tag referring to the end-diastolic instant marking the end of the cycle.
    """

    ED_E: str = "ED_E"


@dataclass(frozen=True)
class CamusTags(Tags):
    """Class to gather the tags referring to CAMUS specific data, from both the training and result datasets.

    Args:
        registered: Tag indicating whether the dataset was registered.
        full_sequence: Tag indicating whether the dataset contains complete sequence between ED and ES for each view.
        instants: Tag indicating the clinically important instants available in the sequence.
        img_proc: Tag referring to resized images, used as input when training models.
        gt_proc: Tag referring to resized groundtruths used as reference when training models.
        info: Tag referring to images' metadata.
        voxelspacing: Tag referring to voxels' size along each (time, height, width) dimension (in mm).
        raw: Tag referring to data before it was processed by some algorithm (e.g. groundtruths before resizing,
            predicted segmentations before post-processing, etc.)
        rec: Tag referring to data that was reconstructed by an autoencoder model.
        frame_pos: Tag referring to the frame normalized index in the sequence (normalized so that ED=0 and ES=1).
    """

    registered: str = "register"
    full_sequence: str = "sequence"
    instants: str = "instants"

    img_proc: str = "img_proc"
    gt_proc: str = "gt_proc"
    info: str = "info"
    voxelspacing: str = "voxelspacing"

    raw: str = "raw"
    rec: str = "rec"

    frame_pos: str = "frame_pos"


in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the ultrasound image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
