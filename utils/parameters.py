from dataclasses import dataclass
from typing import Tuple, Sequence, Union

parameters = dataclass(frozen=True)


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
    max_nb_epochs: int
    gpus: Union[int, Sequence[int]]
    nb_gpu_nodes: int
