import sys
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchinfo import summary

from vital.data.config import DataParameters


class VitalSystem(pl.LightningModule, ABC):
    """Top-level abstract system from which to inherit.

    Implementations of behaviors related to each phase of training (e.g. data preparation, training, evaluation) are
    made through mixins for this class.

    Implements useful generic utilities and boilerplate Lighting code:
        - CLI for generic arguments
    """

    def __init__(self, optim: DictConfig, data_params: DataParameters, **kwargs):
        """Saves the parameters from all the model's childs and mixins in `hparams`.

        Args:
            optim: hydra configuration for the optimizer.
            data_params: Parameters related to the data necessary to initialize the model.
            **kwargs: Dictionary of arguments to save as the model's `hparams`.
        """
        super().__init__()
        # Collection of hyperparameters configuring the system
        self.save_hyperparameters()

        # By default, assumes the provided data shape is in channel-first format
        self.example_input_array = torch.randn((2, *self.hparams.data_params.in_shape))

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        self.log_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        # Save Keras-style summary to a ``summary.txt`` file, inside the output directory
        # Option to disable the summary if PL model's summary is disabled to avoid possible device incompatibilities
        # (e.g. in clusters).
        if self.hparams.enable_model_summary and self.global_rank == 0:
            model_summary = summary(
                self,
                input_data=self.example_input_array,
                col_names=["input_size", "output_size", "kernel_size", "num_params"],
                depth=sys.maxsize,
                device=self.device,
                verbose=0,
            )
            (self.log_dir / "summary.txt").write_text(str(model_summary), encoding="utf-8")

    @property
    def log_dir(self) -> Path:
        """Returns the root directory where test logs get saved."""
        return Path(self.trainer.log_dir) if self.trainer.log_dir else Path(self.hparams.trainer.default_root_dir)

    def configure_optimizers(self) -> Optimizer:  # noqa: D102
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())


class SystemComputationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the training/validation/testing phases."""

    def __init__(self, train_log_kwargs: dict, val_log_kwargs: dict, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.train_log_kwargs = train_log_kwargs
        self.val_log_kwargs = val_log_kwargs

    def training_step(self, *args, **kwargs) -> Union[Tensor, Dict[Union[Literal["loss"], Any], Any]]:  # noqa: D102
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):  # noqa: D102
        pass


class SystemEvaluationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the evaluation phase."""

    def test_step(self, *args, **kwargs):  # noqa: D102
        pass

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """Runs at the end of a test epoch with the output of all test steps.

        It can be used to export results using custom loggers, while not returning any metrics to display in the
        progress bar (as Lightning usually expects).
        """
        pass
