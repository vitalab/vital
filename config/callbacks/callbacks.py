from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Union
from typing import Optional

from omegaconf import MISSING


@dataclass
class CallbacksConf:
    _target_: str = MISSING


@dataclass
class EarlyStoppingConf(CallbacksConf):
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = 'early_stop_on'
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = 'min'
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: bool = False


@dataclass
class ModelCheckpointConf(CallbacksConf):
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: Optional[Any] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: Optional[int] = None
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    every_n_val_epochs: Optional[int] = None
    period: Optional[int] = None


default_callbacks = [
    {"model_checkpoint": ModelCheckpointConf},
    {"early_stopping": EarlyStoppingConf}
]
