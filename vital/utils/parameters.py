from dataclasses import dataclass
from typing import Tuple

parameters = dataclass(frozen=True)


@parameters
class DataParameters:
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
