from dataclasses import dataclass

from config.modules.network import NetworkConfig
from omegaconf import MISSING


@dataclass
class UnetConfig(NetworkConfig):
    _target_: str = "vital.modules.segmentation.unet.UNet"

    init_channels: int = 32
    use_batchnorm: bool = True
    bilinear: bool = False
    dropout_prob: float = 0.0
