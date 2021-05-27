from dataclasses import dataclass

from config.conf import DefaultConfig
from config.data.acdc import AcdcConfig
from config.data.data import DataConfig
from config.data.mnist import MnistConfig
from config.system.classification import ClassificationConfig
from config.system.modules.mlp import MLPConfig
from config.system.modules.module import ModuleConfig
from config.system.modules.unet import UNetConfig
from config.system.segmentation import SegmentationConfig
from config.system.system import SystemConfig


@dataclass
class AcdcUNetConfig(DefaultConfig):
    data: DataConfig = AcdcConfig
    system: SystemConfig = SegmentationConfig
    module: ModuleConfig = UNetConfig


@dataclass
class MnistMLPConfig(DefaultConfig):
    data: DataConfig = MnistConfig
    system: SystemConfig = ClassificationConfig
    module: ModuleConfig = MLPConfig
