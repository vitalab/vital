from collections import Sequence
from dataclasses import dataclass

from config.system.modules.module import ModuleConfig


@dataclass
class MLPConfig(ModuleConfig):
    _target_: str = "vital.modules.classification.mlp.MLP"

    hidden: Sequence[int] = (128,)
    # output_activation: Optional[nn.Module] = None
    dropout_rate: float = 0.25
