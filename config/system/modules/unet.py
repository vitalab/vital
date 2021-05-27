from dataclasses import dataclass

from config.system.modules.module import ModuleConfig


@dataclass
class UNetConfig(ModuleConfig):
    _target_: str = "vital.modules.segmentation.unet.UNet"

    init_channels: int = 32
    use_batchnorm: bool = True
    bilinear: bool = False
    dropout_prob: float = 0.0
