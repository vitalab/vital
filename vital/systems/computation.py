from abc import ABC
from argparse import ArgumentParser
from typing import Dict

from pytorch_lightning import TrainResult
from pytorch_lightning.core.step_result import EvalResult
from torch import Tensor

from vital.systems.vital_system import SystemComputationMixin
from vital.utils.format import prefix


class SupervisedComputationMixin(SystemComputationMixin, ABC):
    """Abstract mixin for generic supervised train/val step.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    _logging_flags = ["on_step", "on_epoch", "logger", "prog_bar"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_logging_flags = {flag: flag in self.hparams.train_logging_flags for flag in self._logging_flags}
        self._val_logging_flags = {flag: flag in self.hparams.val_logging_flags for flag in self._logging_flags}

    def trainval_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Handles steps for both training and validation loops, assuming the behavior should be the same.

        For models where the behavior in training and validation is different, then override ``training_step`` and
        ``validation_step`` directly (in which case ``trainval_step`` doesn't need to be implemented).
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> TrainResult:  # noqa: D102
        output = prefix(self.trainval_step(*args, **kwargs), "train_")
        result = TrainResult(minimize=output["train_loss"])
        result.log_dict(output, **self._train_logging_flags)
        return result

    def validation_step(self, *args, **kwargs) -> EvalResult:  # noqa: D102
        output = prefix(self.trainval_step(*args, **kwargs), "val_")
        result = EvalResult(checkpoint_on=output["val_loss"])
        result.log_dict(output, **self._val_logging_flags)
        return result

    @classmethod
    def add_computation_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        parser = super().add_computation_args(parser)
        parser.add_argument(
            "--train_logging_flags",
            type=str,
            nargs="+",
            choices=cls._logging_flags,
            default=["on_step", "logger"],
            help="Options to use for logging the training metrics. \nThe options apply to all training metrics)",
        )
        parser.add_argument(
            "--val_logging_flags",
            type=str,
            nargs="+",
            choices=cls._logging_flags,
            default=["on_epoch", "logger"],
            help="Options to use for logging the validation metrics. \nThe options apply to all validation metrics)",
        )
        return parser
