from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

import numpy as np
from torch import Tensor


@dataclass
class PatientData:
    """Data structure that bundles data from the ACDC dataset for one patient.

    Args:
        id: patient's identifier (in format "patient0123").
    """

    id: str
    instants: MutableMapping[str, "InstantData"] = field(default_factory=dict)


@dataclass
class InstantData:
    """Data structure that bundles data from the ACDC dataset for one Instant (ED or ES).

    Args:
        id: patient's identifier (in format "patient0123").
    """

    img: Tensor
    gt: Tensor
    voxelspacing: np.ndarray
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)
