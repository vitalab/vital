import os
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    _target_: str = MISSING
    batch_size: int = 32
    num_workers: int = os.cpu_count() - 1
