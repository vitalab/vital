from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from vital.data.camus.config import Instant, View


@dataclass
class PatientData:
    """
    Args:
        id: patient's identifier (in format "patient0123").
        views: mapping between each view available for the patient and the data associated with the view.
    """
    id: str
    views: Mapping[View, "ViewData"] = field(default_factory=dict)


@dataclass
class ViewData:
    """
    Args:
        gt: unprocessed groundtruths, used as reference when evaluating models' scores.
        info: images' metadata.
        instants_with_gt: mapping between instants that have manually annotated segmentations and their indices in the
                          view.
        registering: parameters applied originally to register the images and groundtruths.
    """
    gt: np.ndarray
    info: np.ndarray
    instants_with_gt: Mapping[Instant, int]
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)