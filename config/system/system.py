from collections import Callable
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class SystemConfig:
    _target_: str = MISSING
