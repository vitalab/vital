from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

import numpy as np


@dataclass
class PatientData:
    """Collection of data relevant to a patient (split across multiple views).

    Args:
        id: Patient's identifier (in format "patient0123").
        views: Mapping between each view available for the patient and the data associated with the view.
    """

    id: str
    views: MutableMapping[str, "ViewData"] = field(default_factory=dict)


@dataclass
class ViewData:
    """Collection of data relevant to a specific view sequence.

    Args:
        gt: Unprocessed groundtruths, used as reference when evaluating models' scores.
        info: Images' metadata.
        instants: Mapping between instant IDs and their frame index in the view.
        registering: Parameters applied originally to register the images and groundtruths.
    """

    gt: np.ndarray
    info: np.ndarray
    instants: Mapping[str, int]
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)
