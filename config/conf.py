from dataclasses import dataclass, field
from typing import Optional, Union, Any, List, Dict

import torch
from config.callbacks.callbacks import CallbacksConf, default_callbacks
from config.data.data import DataConfig
from config.loggers.logger import LoggerConfig
from config.system.modules.module import ModuleConfig
from config.system.system import SystemConfig
from omegaconf import MISSING


@dataclass
class TrainerConfig:
    _target_: str = 'pytorch_lightning.Trainer'
    fast_dev_run: bool = False
    default_root_dir: Optional[str] = None
    gpus: int = int(torch.cuda.is_available())
    # max_epochs: int = 300


@dataclass
class DefaultConfig:
    system: SystemConfig = MISSING
    module: ModuleConfig = MISSING
    data: DataConfig = MISSING
    trainer: TrainerConfig = TrainerConfig()

    callbacks: Dict = field(default_factory=lambda: default_callbacks)

    seed: Optional[int] = None

    resume: bool = False
    ckpt_path: Optional[str] = None

    logger: Any = True  # True defaults to TensorboardLogger in PL trainer

    train: bool = True
    test: bool = True
