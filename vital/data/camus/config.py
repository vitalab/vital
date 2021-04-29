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


image_size: int = 256
"""Dimension of the images in the dataset (in pixels)."""

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the ultrasound image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""

camus_100_patients = [
'patient0002',
'patient0004',
'patient0005',
'patient0008',
'patient0010',
'patient0012',
'patient0013',
'patient0014',
'patient0016',
'patient0017',
'patient0018',
'patient0019',
'patient0020',
'patient0021',
'patient0022',
'patient0023',
'patient0024',
'patient0025',
'patient0027',
'patient0028',
'patient0029',
'patient0031',
'patient0034',
'patient0035',
'patient0036',
'patient0038',
'patient0040',
'patient0041',
'patient0043',
'patient0044',
'patient0046',
'patient0047',
'patient0049',
'patient0051',
'patient0053',
'patient0056',
'patient0057',
'patient0059',
'patient0060',
'patient0061',
'patient0063',
'patient0064',
'patient0066',
'patient0069',
'patient0071',
'patient0072',
'patient0073',
'patient0074',
'patient0076',
'patient0080',
'patient0082',
'patient0083',
'patient0085',
'patient0088',
'patient0091',
'patient0092',
'patient0093',
'patient0094',
'patient0097',
'patient0100',
'patient0101',
'patient0103',
'patient0104',
'patient0107',
'patient0108',
'patient0110',
'patient0111',
'patient0112',
'patient0113',
'patient0115',
'patient0116',
'patient0117',
'patient0118',
'patient0120',
'patient0122',
'patient0123',
'patient0124',
'patient0125',
'patient0126',
'patient0128',
'patient0129',
'patient0133',
'patient0136',
'patient0138',
'patient0139',
'patient0140',
'patient0141',
'patient0142',
'patient0143',
'patient0144',
'patient0146',
'patient0148',
'patient0149',
'patient0150',
'patient0151',
'patient0152',
'patient0156',
'patient0228'
]