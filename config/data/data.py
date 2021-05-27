import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    _target_: str = "vital.data.acdc.data_module.AcdcDataModule"
    batch_size: int = 32
    num_workers: int = os.cpu_count() - 1
