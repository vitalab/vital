from abc import ABC
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from typing import Tuple, Mapping, Union, List, Dict

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import VisionDataset

from vital.data.config import Subset
from vital.utils.parameters import DataParameters


class VitalSystem(pl.LightningModule, ABC):
    """Top-level abstract system from which to inherit.

    Implementations of behaviors related to each phase of training (e.g. data preparation, training, evaluation) are
    made through mixins for this class.

    Implements useful generic utilities and boilerplate Lighting code:
        - CLI for generic arguments
    """

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.module: nn.Module  # field in which to assign the network used by the implementation of ``System``

        # To initialize in ``DataManagerMixin``
        self.data_params: DataParameters
        self.dataset: Mapping[Subset, VisionDataset]

    def save_model_summary(self, system_input_shape: Tuple[int, ...]):
        """Saves a summary of the model in a format similar to Keras' summary.

        Args:
            system_input_shape: shape of the input data the system should expect when using the dataset.
        """
        with open(str(self.hparams.default_root_dir.joinpath('summary.txt')), 'w') as f:
            with redirect_stdout(f):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                summary(self.module.to(device), system_input_shape)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def forward(self, *args, **kwargs):
        return self.module.predict(*args, **kwargs)

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Builds a parser object that supports CL arguments specific to a system.

        Must be overridden to add generic arguments whose default values are implementation specific (listed below).
            - batch_size
            - lr (if using the default optimizer)
            - weight_decay (if using the default optimizer)

        Also where new system specific arguments should be added (and parsed in the same class' ``parse_args``).
        Generic arguments with model specific values:

        Returns:
            parser object that supports CL arguments specific to a system.
        """
        parser = ArgumentParser(add_help=False)
        parser = cls.add_data_manager_args(parser)
        parser = cls.add_train_eval_loop_args(parser)
        parser = cls.add_evaluation_args(parser)
        return parser


class SystemDataManagerMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the interface between the Datasets and DataLoaders."""
    data_params: DataParameters
    dataset: Mapping[Subset, VisionDataset]

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass

    @classmethod
    def add_data_manager_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds data related arguments to a parser object.

        Args:
            parser: parser object to which to add data loop related arguments.

        Returns:
            parser object to which data loop related arguments have been added.
        """
        return parser


class SystemTrainEvalLoopMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the training/validation/testing phases."""

    def training_step(self, *args, **kwargs) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        raise NotImplementedError

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        pass

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Expects the system to make test-time predictions for the batch, coordinating with ``save_test_step_results``
        from the ``SystemEvaluationMixin`` to save the results.

        It doesn't have to output any metric for downstream tasks, like Lightning normally expects.
        """
        pass

    @classmethod
    def add_train_eval_loop_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds train-eval loop related arguments to a parser object.

        Args:
            parser: parser object to which to add train-eval loop related arguments.

        Returns:
            parser object to which train-eval loop related arguments have been added.
        """
        return parser


class SystemEvaluationMixin(VitalSystem, ABC):
    """``VitalSystem`` mixin for handling the evaluation phase."""

    def save_test_step_results(self, batch_idx: int, **kwargs):
        """Saves predictions made by ``test_step`` from the ``SystemTrainEvalLoopMixin`` and additional relevant
        information the downstream evaluation."""
        pass

    def test_epoch_end(self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]) \
            -> Dict[str, Dict[str, Tensor]]:
        """Called at the end of the testing epoch. Can be used to export results using custom loggers, while not
        returning any metrics to display in the progress bar (as Lightning normally expects).
        """
        pass

    @classmethod
    def add_evaluation_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds evaluation related arguments to a parser object.

        Args:
            parser: parser object to which to add evaluation related arguments.

        Returns:
            parser object to which evaluation related arguments have been added.
        """
        return parser
