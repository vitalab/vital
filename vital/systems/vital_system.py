from abc import ABC
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Literal, Mapping, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from torch import Tensor, nn
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

    # Fields to initialize in implementation of ``VitalSystem``
    #: Network instance called by ``VitalSystem`` for computations
    module: nn.Module

    # Fields to initialize in implementation of ``DataManagerMixin``
    #: Collection of parameters related to the nature of the data
    data_params: DataParameters
    #: Mapping between subsets of the data (e.g. train) and their torch ``Dataset`` handle
    dataset: Mapping[Subset, Dataset]

    def __init__(self, hparams: Namespace):
        super().__init__()
        #: Collection of hyperparameters configuring the system
        self.hparams = hparams

        # Ensure output directory exists
        self.hparams.default_root_dir.mkdir(parents=True, exist_ok=True)

    def save_model_summary(self, system_input_shape: Tuple[int, ...]) -> None:
        """Saves a summary of the model in a format similar to Keras' summary.

        Will not be printed if PL weights' summary was disable.
        This is done to avoid possible device incompatibilities in clusters.

        Args:
            system_input_shape: Shape of the input data the system should expect when using the dataset.
        """
        with open(str(self.hparams.default_root_dir.joinpath("summary.txt")), "w") as f:
            summary_str, _ = summary_info(self.module, system_input_shape, self.device)
            f.write(summary_str)

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

    @auto_move_data
    def forward(self, *args, **kwargs):  # noqa: D102
        return self.module(*args, **kwargs)

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
