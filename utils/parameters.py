from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Sequence, Union

parameters = dataclass(frozen=True)


@parameters
class SystemParameters:
    save_to: Path
    pretrained: Path = None


@parameters
class DataShape:
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]


@parameters
class DataParameters:
    shape: DataShape
    use_da: bool
    batch_size: int
    workers: int


@parameters
class OptimizerParameters:
    lr: float


@parameters
class TrainerParameters:
    default_save_path: str
    fast_dev_run: bool
    weights_summary: str
    min_epochs: int
    max_epochs: int
    gpus: Union[int, Sequence[int]]
    num_nodes: int
