from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

import numpy as np

from vital.data.camus.config import Instant, View


@dataclass
class PatientData:
    """Collection of data relevant to a patient (split across multiple views).

    Args:
        id: Patient's identifier (in format "patient0123").
        views: Mapping between each view available for the patient and the data associated with the view.
    """

    id: str
    views: MutableMapping[View, "ViewData"] = field(default_factory=dict)


@dataclass
class ViewData:
    """Collection of data relevant to a specific view sequence.

    Args:
        gt: Unprocessed groundtruths, used as reference when evaluating models' scores.
        info: Images' metadata.
        instants_with_gt: Mapping between instants that have manually annotated segmentations and their indices in the
            view.
        registering: Parameters applied originally to register the images and groundtruths.
    """

    gt: np.ndarray
    info: np.ndarray
    instants_with_gt: Mapping[Instant, int]
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)
