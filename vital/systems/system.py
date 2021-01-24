import os
import sys
from abc import ABC
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.memory import ModelSummary
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from vital.data.config import DataParameters, Subset


class VitalSystem(pl.LightningModule, ABC):
    """Top-level abstract system from which to inherit.

    Implementations of behaviors related to each phase of training (e.g. data preparation, training, evaluation) are
    made through mixins for this class.

    Implements useful generic utilities and boilerplate Lighting code:
        - CLI for generic arguments
    """

    # Fields to initialize in implementation of ``DataManagerMixin``
    #: Collection of parameters related to the nature of the data
    data_params: DataParameters
    #: Mapping between subsets of the data (e.g. train) and their torch ``Dataset`` handle
    dataset: Mapping[Subset, Dataset]

    def __init__(self, **kwargs):
        """Saves the parameters from all the model's childs and mixins in `hparams`.

        Args:
            **kwargs: Dictionary of arguments to save as the model's `hparams`.
        """
        super().__init__()
        # Collection of hyperparameters configuring the system
        self.save_hyperparameters()

        # By default, assumes the provided data shape is in channel-first format
        self.example_input_array = torch.randn((self.hparams.batch_size, *self.hparams.data_params.in_shape))

    def on_pretrain_routine_start(self) -> None:  # noqa: D102
        self.log_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    @property
    def log_dir(self) -> Path:
        """Returns the root directory where test logs get saved."""
        return Path(self.trainer.log_dir) if self.trainer.log_dir else self.hparams.default_root_dir

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        """Adds saving a Keras-style summary of the model to the base PL summary routine.

        The Keras-style summary is saved to a ``summary.txt`` file, inside the output directory.

        Notes:
            - Requires the ``example_input_array`` property to be set for the module.
            - Will not be printed if PL weights' summary was disabled (``mode == None``). This is done to avoid possible
              device incompatibilities in clusters.
        """
        if mode is not None:
            model_summary = summary(
                self,
                input_data=self.example_input_array,
                verbose=0,
                depth=sys.maxsize,
                col_names=["input_size", "output_size", "kernel_size", "num_params"],
            )
            (self.log_dir / "summary.txt").write_text(str(model_summary), encoding="utf-8")
        return super().summarize(mode)

    def configure_optimizers(self) -> Optimizer:  # noqa: D102
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Builds a parser object that supports command line arguments specific to a system.

        Must be overridden to add generic arguments whose default values are implementation specific (listed below).
            - 'batch_size'
            - 'lr' (if using the default optimizer)
            - 'weight_decay' (if using the default optimizer)

        Also where new system specific arguments should be added (and parsed in the same class' ``parse_args``).
        Generic arguments with model specific values:

        Returns:
            Parser object that supports command line arguments specific to a system.
        """
        parser = ArgumentParser(add_help=False)
        return cls.add_evaluation_args(cls.add_computation_args(cls.add_data_manager_args(parser)))


class SystemDataManagerMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the interface between the `Datasets` and `DataLoaders`."""

    data_params: DataParameters
    dataset: Mapping[Subset, Dataset]

    def prepare_data(self) -> None:
        """Runs data setup might write to disk or that need to be done only from a single GPU in distributed settings.

        This is the ideal place to download the dataset.
        """
        pass

    def setup(self, stage: Literal["fit", "test"]) -> None:
        """Runs data setup you might want to perform on every GPU, e.g. assigning the state.

        This is the ideal place to initialize the ``Dataset`` instances.
        """
        pass

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        raise NotImplementedError

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:  # noqa: D102
        pass

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:  # noqa: D102
        pass

    @classmethod
    def add_data_manager_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds data related arguments to a parser object.

        Args:
            parser: Parser object to which to add data loop related arguments.

        Returns:
            Parser object to which data loop related arguments have been added.
        """
        parser.add_argument(
            "--num_workers",
            type=int,
            default=os.cpu_count() - 1,
            help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main "
            "process.",
        )
        return parser


class SystemComputationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the training/validation/testing phases."""

    #: Choice of logging flags to toggle through the CLI
    _logging_flags = ["on_step", "on_epoch", "logger", "prog_bar"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Update logging flags to map between available flags and their boolean values,
        # instead of listing desired flags
        self.train_log_kwargs = {flag: (flag in self.hparams.train_logging_flags) for flag in self._logging_flags}
        self.val_log_kwargs = {flag: (flag in self.hparams.val_logging_flags) for flag in self._logging_flags}

    def training_step(self, *args, **kwargs) -> Union[Tensor, Dict[Union[Literal["loss"], Any], Any]]:  # noqa: D102
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):  # noqa: D102
        pass

    @classmethod
    def add_computation_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds computation (train/val step and test-time inference) related arguments to a parser object.

        Args:
            parser: Parser object to which to add train-eval loop related arguments.

        Returns:
            Parser object to which computation related arguments have been added.
        """
        parser.add_argument(
            "--train_logging_flags",
            type=str,
            nargs="+",
            choices=cls._logging_flags,
            default=["on_step", "logger"],
            help="Options to use for logging the training metrics. The options apply to all training metrics",
        )
        parser.add_argument(
            "--val_logging_flags",
            type=str,
            nargs="+",
            choices=cls._logging_flags,
            default=["on_epoch", "logger"],
            help="Options to use for logging the validation metrics. The options apply to all validation metrics",
        )
        return parser


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

    @classmethod
    def add_evaluation_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds evaluation related arguments to a parser object.

        Args:
            parser: Parser object to which to add evaluation related arguments.

        Returns:
            Parser object to which evaluation related arguments have been added.
        """
        return parser
