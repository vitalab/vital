from dataclasses import dataclass

from config.system.modules.module import ModuleConfig


@dataclass
class EnetConfig(ModuleConfig):
    _target_: str = "vital.modules.segmentation.enet.Enet"

    init_channels: int = 16
    dropout: float = 0.1
    encoder_relu: bool = True
    decoder_relu: bool = True
