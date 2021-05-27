from dataclasses import dataclass

from config.system.system import SystemConfig


@dataclass
class SupervisedConfig(SystemConfig):
    _target_: str = "vital.systems.supervised.SupervisedComputationMixin"
