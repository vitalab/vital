from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping, Tuple

import numpy as np


@dataclass
class ViewMetadata:
    """Collection of data relevant to a specific view sequence.

    Args:
        gt: Ground truths at their native resolution, used as reference when evaluating models' scores.
        voxelspacing: Size of the segmentations' voxels along each (time, height, width) dimension (in mm).
        instants: Mapping between instant IDs and their frame index in the view.
        registering: Parameters to register the groundtruths.
    """

    gt: np.ndarray
    voxelspacing: Tuple[Real, Real, Real]
    instants: Mapping[str, int]
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)
