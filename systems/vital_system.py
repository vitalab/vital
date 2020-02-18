from abc import ABC
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Union, Tuple, Mapping, Literal

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from vital.utils.config import Subset
from vital.utils.parameters import parameters, OptimizerParameters, TrainerParameters, DataParameters, SystemParameters


class VitalSystem(pl.LightningModule, ABC):
    """Abstract system from which project's base systems should inherit.

    Implements useful generic utilities and boilerplate Lighting code:
        - CLI for generic arguments
        - Interface between the Datasets and DataLoaders
        - Handling of train/val step results (metrics logging and printing)
    """
    use_da: bool = False  # whether the system applies Data Augmentation (DA) by default.

    def __init__(self, system_params: SystemParameters,
                 data_params: DataParameters,
                 optim_params: OptimizerParameters,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_params = data_params
        self.optim_params = optim_params
        self.system_params = system_params
        self.dataset: Mapping[Subset, Dataset]  # subsets of the dataset used to configure train/val/test DataLoaders
        self.module: nn.Module  # field in which to assign the network used by the implementation of ``System``

        system_params.save_to.mkdir(parents=True, exist_ok=True)

    def save_model_summary(self, system_input_shape: Tuple[int, ...]):
        """ Saves a summary of the model in a format similar to Keras' summary.

        Args:
            system_input_shape: shape of the input data the system should expect when using the dataset.
        """
        with open(str(self.system_params.save_to.joinpath('summary.txt')), 'w') as f:
            with redirect_stdout(f):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                summary(self.module.to(device), system_input_shape)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **asdict(self.optim_params))

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset[Subset.TRAIN],
                          batch_size=self.data_params.batch_size, shuffle=True,
                          num_workers=self.data_params.workers, pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset[Subset.VALID],
                          batch_size=self.data_params.batch_size, shuffle=True,
                          num_workers=self.data_params.workers, pin_memory=True)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset[Subset.TEST], batch_size=None, num_workers=self.data_params.workers)

    def trainval_step(self, batch, batch_idx, metric_prefix: Literal['', 'val_'] = ''):
        """Must be implemented if either ``training_step`` or ``validation_step`` are not overridden.

        As the name indicates, handles steps for both training and validation loops, assuming the behavior should be
        the same. For models where the behavior in training and validation is different, than override
        ``training_step`` and ``validation_step`` directly."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Returns loss and metrics for callbacks, assuming ``training_end`` is not overridden."""
        metrics = self.trainval_step(batch, batch_idx)
        loss = metrics['loss']
        return {'loss': loss,
                'progress_bar': metrics,
                'log': metrics}

    def validation_step(self, batch, batch_idx):
        """Returns metrics to be reduced ànd made accessible for display and callbacks in ``validation_end`."""
        return self.trainval_step(batch, batch_idx, metric_prefix='val_')

    def validation_end(self, outputs):
        """By default, mark all metrics for ``progress_bar`` and ``log``."""
        metric_names = outputs[0].keys()
        reduced_metrics = {metric_name: torch.stack([output[metric_name]
                                                     for output in outputs]).mean()
                           for metric_name in metric_names}
        return {'progress_bar': reduced_metrics,
                'log': reduced_metrics}

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """ Builds a parser object that supports generic CL arguments used by ``VitalSystem``s.

        Must be overridden to add generic arguments whose default values are model specific (listed below).
        Also where new system specific arguments should be added (and parsed in the same class' ``parse_args``).
        Generic arguments with model specific values:
            - lr
            - batch_size
            - max_epochs

        Returns:
            parser object that supports generic CL arguments used by ``VitalSystem``s.
        """
        parser = ArgumentParser(add_help=False)

        if cls.use_da:
            parser.add_argument("no_da", dest="use_da", action='store_false', help="Disable dataset augmentation")
        else:
            parser.add_argument("--use_da", dest="use_da", action='store_true', help="Enable dataset augmentation")

        # save/load parameters
        parser.add_argument("--save_to", type=Path, help="Path for logs+weights+results")
        parser.add_argument("--pretrained", type=Path, help="Path to Lightning module checkpoints to restore system")

        # evaluation parameters
        parser.add_argument("--predict", action='store_true', help="Skip training and do test phase")

        # training configuration parameters
        parser.add_argument('--weights_summary', type=str, default='full', choices=['full', 'top'])
        parser.add_argument('--gpus', type=Union[int, List[int]], default=1)
        parser.add_argument('--num_nodes', type=int, default=1)
        parser.add_argument('--workers', type=int, default=0)

        # Lightning configuration parameters
        parser.add_argument('--fast_dev_run', action='store_true',
                            help="Runs full iteration over everything to find bugs")
        return parser

    @classmethod
    def parse_args(cls, args: Namespace) -> Dict[str, parameters]:
        """ Bundles CL arguments into collections of related parameters.

        Must be overridden to parse ``DataParameters``, since no generic ``DataShape`` can be inferred to provide a
        sane default value.
        Also where to parse any system specific arguments (added in the same class' ``build_parser``).

        Args:
            args: arguments parsed from the system trainer's CLI.

        Returns:
            collections of related parameters.
        """
        trainer_params = TrainerParameters(default_save_path=args.save_to,
                                           fast_dev_run=args.fast_dev_run,
                                           weights_summary=args.weights_summary,
                                           gpus=args.gpus,
                                           num_nodes=args.num_nodes,
                                           min_epochs=args.min_epochs if 'min_epochs' in args else 1,
                                           max_epochs=args.max_epochs)
        optim_params = OptimizerParameters(lr=args.lr)
        system_params = SystemParameters(
            save_to=args.save_to,
            pretrained=args.pretrained,
        )
        return {'trainer_params': trainer_params,
                'optim_params': optim_params,
                'system_params': system_params,
                'predict': args.predict}
