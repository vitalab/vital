from dataclasses import dataclass
from typing import Optional, Union, Any

import torch
from config.data.data import DataConfig
from config.system.modules.module import ModuleConfig
from config.system.system import SystemConfig
from omegaconf import MISSING


@dataclass
class TrainerConfig:
    _target_: str = 'pytorch_lightning.Trainer'
    fast_dev_run: bool = False
    default_root_dir: Optional[str] = None
    gpus: int = int(torch.cuda.is_available())
    max_epochs: int = 300


@dataclass
class LoggerConfig:
    _target_: str = MISSING


@dataclass
class CometConfig:
    _target_: str = 'pytorch_lightning.loggers.CometLogger'


@dataclass
class DefaultConfig:
    system: SystemConfig = MISSING
    module: ModuleConfig = MISSING
    data: DataConfig = MISSING
    trainer: TrainerConfig = TrainerConfig()

    seed: Optional[int] = None

    resume: bool = False
    ckpt_path: Optional[str] = None

    logger: Any = True

    train: bool = True
    test: bool = True
