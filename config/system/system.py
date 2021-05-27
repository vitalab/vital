from dataclasses import dataclass

from config.system.modules.module import ModuleConfig
from omegaconf import MISSING


@dataclass
class SystemConfig:
    _target_: str = MISSING
    # module: ModuleConfig = MISSING
