from dataclasses import dataclass

from config.system.modules.mlp import MLPConfig
from config.system.system import SystemConfig


@dataclass
class ClassificationConfig(SystemConfig):
    _target_: str = "vital.systems.classification.ClassificationComputationMixin"
    # module = MLPConfig
