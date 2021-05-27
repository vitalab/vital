from dataclasses import dataclass

from config.data.data import DataConfig


@dataclass
class MnistConfig(DataConfig):
    _target_: str = "vital.data.mnist.data_module.MnistDataModule"

    dataset_path: str = "/tmp"
    transform = None
    target_transform = None
    download: bool = True
