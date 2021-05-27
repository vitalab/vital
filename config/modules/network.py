from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class NetworkConfig:
    _target_: str = MISSING
