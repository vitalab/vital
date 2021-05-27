from collections import Callable
from dataclasses import MISSING, dataclass

from config.system.system import SystemConfig


@dataclass
class SupervisedConfig(SystemConfig):
    _target_: str = "vital.systems.supervised.SupervisedComputationMixin"
