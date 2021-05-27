from dataclasses import dataclass

from config.system.modules.mlp import MLPConfig
from config.system.system import SystemConfig


@dataclass
class SegmentationConfig(SystemConfig):
    _target_: str = "vital.systems.segmentation.SegmentationComputationMixin"
    # module = MISSING

