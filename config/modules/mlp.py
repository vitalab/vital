from collections import Sequence
from dataclasses import dataclass
from typing import Optional

from config.modules.network import NetworkConfig
from torch import nn


@dataclass
class MLPConfig(NetworkConfig):
    _target_: str = "vital.modules.classification.mlp.MLP"

    hidden: Sequence[int] = (128,)
    # output_activation: Optional[nn.Module] = None
    dropout_rate: float = 0.25
