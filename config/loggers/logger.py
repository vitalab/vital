from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class LoggerConfig:
    _target_: str = MISSING


@dataclass
class CometConfig:
    _target_: str = 'pytorch_lightning.loggers.CometLogger'
