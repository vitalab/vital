from dataclasses import dataclass
from typing import Tuple

parameters = dataclass(frozen=True)


@parameters
class DataParameters:
    """Class for defining parameters related to the nature of the data.

    Args:
        in_shape: shape of the input data (e.g. height, width, channels).
        out_shape: shape of the target data (e.g. height, width, channels).
    """

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
