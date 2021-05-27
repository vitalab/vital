from collections import Callable
from dataclasses import dataclass

from config.system.supervised import SupervisedConfig


@dataclass
class LossConfig:
    _target_: str = "vital.utils.loss.segmentation.DiceCELoss"


@dataclass
class SegmentationConfig(SupervisedConfig):
    loss: LossConfig = LossConfig()
