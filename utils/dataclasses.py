from dataclasses import dataclass
from typing import Tuple

parameters = dataclass(frozen=True)


@parameters
class DataParameters:
    batch_size: int
    use_da: bool


@parameters
class DataShape:
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]


@parameters
class OptimizerParameters:
    lr: float


@parameters
class TrainerParameters:
    default_save_path: str
    fast_dev_run: bool
    max_nb_epochs: int
    gpus: int
    nb_gpu_nodes: int
