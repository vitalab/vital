from dataclasses import dataclass

from config.data.data import DataConfig
from omegaconf import MISSING
import os

@dataclass
class AcdcConfig(DataConfig):
    _target_: str = "vital.data.acdc.data_module.AcdcDataModule"

    # dataset_path: str = os.environ.get('ACDC_DATA_PATH') or MISSING
    dataset_path: str = '$ACDC_DATA_PATH'
    use_da: bool = True
