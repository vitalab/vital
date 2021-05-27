from dataclasses import dataclass

from config.data.data import DataConfig
from omegaconf import MISSING


@dataclass
class AcdcConfig(DataConfig):
    _target_: str = "vital.data.acdc.data_module.AcdcDataModule"

    dataset_path: str = MISSING
    use_da: bool = True
