from dataclasses import dataclass

from config.data.data import DataConfig
from omegaconf import MISSING


@dataclass
class CamusConfig(DataConfig):
    _target_: str = "vital.data.camus.data_module.CamusDataModule"

    dataset_path: str = MISSING
    use_da: bool = True
