import sys
from abc import ABC
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim.optimizer import Optimizer
from torchinfo import summary

from vital.data.config import DataParameters


class VitalSystem(pl.LightningModule, ABC):
    """Top-level abstract system from which to inherit, implementing some generic utilities and boilerplate code.

    Implementations of behaviors related to each phase of training (e.g. data preparation, training, evaluation) should
    be made in specific packages (e.g. `data` for data handling, `tasks` for computation pipeline) or in callbacks
    (e.g. evaluation).
    """

    def __init__(
        self, model: DictConfig, optim: DictConfig, choices: DictConfig, data_params: DataParameters, **kwargs
    ):
        """Saves the system's configuration in `hparams`.

        Args:
            model: Nested Omegaconf object containing the model architecture's configuration.
            optim: Nested Omegaconf object containing the optimizers' configuration.
            choices: Nested Omegaconf object containing the choice made from the pre-set configurations.
            data_params: Parameters related to the data necessary to initialize the model.
            **kwargs: Dictionary of arguments to save as the model's `hparams`.
        """
        super().__init__()
        # Collection of hyperparameters configuring the system
        self.save_hyperparameters()

        # Also save the classpath of the system to be able to load a checkpoint w/o knowing it's type beforehand
        self.save_hyperparameters({"task": {"_target_": f"{self.__class__.__module__}.{self.__class__.__name__}"}})

        # By default, assumes the provided data shape is in channel-first format
        self.example_input_array = torch.randn((2, *self.hparams.data_params.in_shape))

    def configure_model(self) -> nn.Module:
        """Configure the network architecture used by the system."""
        return hydra.utils.instantiate(
            self.hparams.model,
            input_shape=self.hparams.data_params.in_shape,
            output_shape=self.hparams.data_params.out_shape,
        )

    def configure_optimizers(self) -> Optimizer:  # noqa: D102
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())

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
        return Path(self.trainer.log_dir) if self.trainer.log_dir else Path(self.trainer.default_root_dir)
