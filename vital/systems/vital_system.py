from abc import ABC
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Literal, Mapping, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.memory import ModelSummary
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from vital.data.config import Subset
from vital.utils.parameters import DataParameters
from vital.utils.summary import summary_info


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

    def __init__(self, hparams: Union[Dict, Namespace], data_params: DataParameters):  # noqa: D205,D212,D415
        """
        Args:
            hparams: If created straight from CL input, a ``Namespace`` of arguments parsed from the CLI.
                Otherwise (when loaded from checkpoints), a ``Dict`` of deserialized hyperparameters.
            data_params: Provided by the implementation of ``DataManagerMixin`` when it calls its parent's ``__init__``.
        """
        super().__init__()
        #: Collection of hyperparameters configuring the system
        self.hparams = hparams
        self.data_params = data_params

        # Ensure output directory exists
        self.hparams.default_root_dir.mkdir(parents=True, exist_ok=True)

        # By default, assumes the provided data shape is in channel-first format
        self.example_input_array = torch.randn((self.hparams.batch_size, *self.data_params.in_shape))

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        """Adds saving a Keras-style summary of the model to the base PL summary routine.

        The Keras-style summary is saved to a ``summary.txt`` file, inside the ``default_root_dir`` directory.

        Notes:
            - Requires the ``example_input_array`` property to be set for the module.
            - Will not be printed if PL weights' summary was disabled (``mode == None``). This is done to avoid possible
              device incompatibilities in clusters.
        """
        if mode is not None:
            with open(str(self.hparams.default_root_dir.joinpath("summary.txt")), "w") as f:
                summary_str, _ = summary_info(self, self.example_input_array)
                f.write(summary_str)
        return super().summarize(mode)

    def configure_optimizers(self) -> Optimizer:  # noqa: D102
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Builds a parser object that supports CL arguments specific to a system.

        Must be overridden to add generic arguments whose default values are implementation specific (listed below).
            - 'batch_size'
            - 'lr' (if using the default optimizer)
            - 'weight_decay' (if using the default optimizer)

        Also where new system specific arguments should be added (and parsed in the same class' ``parse_args``).
        Generic arguments with model specific values:

        Returns:
            Parser object that supports CL arguments specific to a system.
        """
        parser = ArgumentParser(add_help=False)
        return cls.add_evaluation_args(cls.add_computation_args(cls.add_data_manager_args(parser)))


class SystemDataManagerMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the interface between the `Datasets` and `DataLoaders`."""

    data_params: DataParameters
    dataset: Mapping[Subset, Dataset]

    def prepare_data(self) -> None:  # noqa: D102
        pass

    def setup(self, stage: Literal["fit", "test"]) -> None:
        """Set state to the model before requesting the dataloaders.

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
        return parser


class SystemComputationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the training/validation/testing phases."""

    def training_step(self, *args, **kwargs) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:  # noqa: D102
        raise NotImplementedError

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        pass

    @classmethod
    def add_computation_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds computation (train/val step and test-time inference) related arguments to a parser object.

        Args:
            parser: Parser object to which to add train-eval loop related arguments.

        Returns:
            Parser object to which computation related arguments have been added.
        """
        return parser


class SystemEvaluationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the evaluation phase."""

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        pass

    def test_epoch_end(
        self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
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
