import os
from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Type, Union, List

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from vital.systems.vital_system import VitalSystem


class VitalTrainer(ABC):
    """Abstract trainer that runs the main training/val loop, etc... using Lightning Trainer."""

    @classmethod
    def get_selected_system(cls, hparams: Namespace) -> Type[VitalSystem]:
        """Identify, through the parameters specified in the CLI, the type of the Lightning module chosen by the user.

        Args:
            hparams: arguments parsed from the CLI.

        Returns:
            type of the Lightning module selected by the user to be run.
        """
        raise NotImplementedError

    @classmethod
    def add_non_trainer_args(cls, parser: ArgumentParser):
        """Adds datasets/systems specific subparsers/arguments to a parser object.

        The hierarchy of the added arguments can be arbitrarily complex, as long as ``get_selected_system`` can pinpoint
        a single ``VitalSystem`` to run.

        Args:
            parser: parser object to which non-trainer arguments will be added.
        """
        raise NotImplementedError

    @classmethod
    def main(cls):
        """Sets-up the CLI for the Lightning modules trainable through this trainer and calls the main training/val
        loop.
        """
        # Initialize the parser with generic trainer arguments
        parser = ArgumentParser()
        cls.add_trainer_args(parser)

        # Add subparsers for all systems available through the trainer
        cls.add_non_trainer_args(parser)

        # Parse args and run the target system
        cls.run_system(parser.parse_args())

    @classmethod
    def run_system(cls, hparams: Namespace):
        """Handles the training/validation loop.

        Args:
            hparams: arguments parsed from the CLI.
        """
        early_stop_callback = EarlyStopping(patience=max(1, hparams.max_epochs // 5)) \
            if hparams.early_stop_callback else False
        trainer = Trainer(
            default_save_path=hparams.save_dir,
            fast_dev_run=hparams.fast_dev_run,
            profiler=hparams.profiler,
            weights_summary=hparams.weights_summary,
            gpus=hparams.gpus,
            num_nodes=hparams.num_nodes,
            auto_lr_find=hparams.auto_lr_find,
            auto_scale_batch_size=hparams.auto_scale_batch_size,
            min_epochs=hparams.min_epochs if 'min_epochs' in hparams else 1,
            max_epochs=hparams.max_epochs,
            early_stop_callback=early_stop_callback,
        )

        system_cls = cls.get_selected_system(hparams)

        if hparams.pretrained:  # Load pretrained model if checkpoint is provided
            model = system_cls.load_from_checkpoint(hparams.pretrained)
        else:
            if hparams.predict:  # If we try to skip training and go straight to testing
                raise ValueError("Trainer set to skip training (`--predict` flag) without a pretrained model provided. "
                                 "Please allow model to train (remove `--predict` flag) or "
                                 "provide a pretrained model (through `--pretrained` parameter).")
            else:  # If we have to train and then test the system
                model = system_cls(hparams)
                trainer.fit(model)

        trainer.test(model)

    @classmethod
    def add_trainer_args(cls, parser: ArgumentParser):
        """Adds generic Lightning trainer arguments to a parser object.

        Args:
            parser: parser object to which trainer arguments will be added.
        """
        # save/load parameters
        parser.add_argument("--save_dir", type=Path, required=True, help="Path for logs+weights+results")
        parser.add_argument("--pretrained", type=Path, help="Path to Lightning module checkpoints to restore system")

        # evaluation parameters
        parser.add_argument("--predict", action='store_true', help="Skip training and do test phase")

        # training configuration parameters
        parser.add_argument('--weights_summary', type=str, default=None, choices=['full', 'top'])
        parser.add_argument('--no_early_stopping', dest='early_stop_callback', action='store_false',
                            help="Disable early stopping. \nBy default, early stopping monitors 'val_loss' with a "
                                 "patience of: 'max_epochs' // 5")
        parser.add_argument('--gpus', type=Union[int, List[int]], default=1)
        parser.add_argument('--num_nodes', type=int, default=1)
        parser.add_argument('--workers', type=int, default=os.cpu_count() - 1)

        # Parameters auto-finder
        parser.add_argument('--auto_lr_find', action='store_true')
        parser.add_argument('--auto_scale_batch_size', action='store_true')

        # Lightning configuration parameters
        parser.add_argument('--fast_dev_run', action='store_true',
                            help="Runs full iteration over everything to find bugs")
        parser.add_argument('--profiler', action='store_true',
                            help="Profile standard training events")
        return parser
