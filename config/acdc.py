from dataclasses import dataclass

import torch
from config.data.acdc import AcdcConfig
from config.data.data import DataConfig
from config.modules.network import NetworkConfig
from config.system.system import SystemConfig
from omegaconf import MISSING


@dataclass
class ACDCConfig:
    gpus: int = int(torch.cuda.is_available())

    data: DataConfig = AcdcConfig
    network: NetworkConfig = MISSING
    system: SystemConfig = MISSING
