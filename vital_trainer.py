import logging
import os
from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import copy2
from typing import Type

from pytorch_lightning import Trainer

from vital.systems.vital_system import VitalSystem

logger = logging.getLogger(__name__)


class VitalTrainer(ABC):
    """Abstract trainer that runs the main training/val loop, etc... using Lightning Trainer."""

    @classmethod
    def main(cls):
        """Sets-up the CLI for the Lightning modules trainable through this trainer and calls the main training/val
        loop.
        """
        # Initialize the parser with our own generic arguments
        parser = ArgumentParser()
        parser = cls._add_generic_args(parser)

        # Add Lightning trainer arguments to the parser (overriding some values with our own defaults)
        parser = Trainer.add_argparse_args(parser)

        # Override generic defaults with trainer-specific defaults
        parser = cls._override_trainer_default(parser)

        # Add subparsers for all systems available through the trainer
        parser = cls._add_system_args(parser)

        # Parse args and run the target system
        cls.run_system(cls._parse_and_check_args(parser))

    @classmethod
    def run_system(cls, hparams: Namespace):
        """Handles the training/validation loop.

        Args:
            hparams: arguments parsed from the CLI.
        """
        trainer = Trainer.from_argparse_args(hparams)
        system_cls = cls._get_selected_system(hparams)

        if hparams.pretrained:  # Load pretrained model if checkpoint is provided
            model = system_cls.load_from_checkpoint(hparams.pretrained)
        else:
            if hparams.predict:  # If we try to skip training and go straight to testing
                raise ValueError("Trainer set to skip training (`--predict` flag) without a pretrained model provided. "
                                 "Please allow model to train (remove `--predict` flag) or "
                                 "provide a pretrained model (through `--pretra  ined` parameter).")
            else:  # If we have to train and then test the system
                model = system_cls(hparams)
                trainer.fit(model)

                # Copy best model checkpoint to a fixed path
                best_model_path = cls._define_best_model_save_path(hparams)
                copy2(str(trainer.checkpoint_callback.best_model_path), str(best_model_path))
                logger.info(f"Copied best model checkpoint to: {best_model_path}")

        trainer.test(model)

    @classmethod
    def _define_best_model_save_path(cls, hparams: Namespace) -> Path:
        """Defines the fixed path (w.r.t to the system to run) where to copy the best model checkpoint after training.

        Args:
            hparams: arguments parsed from the CLI.

        Returns:
            fixed path (w.r.t to the system to run) where to copy the best model checkpoint after training.
        """
        system_cls = cls._get_selected_system(hparams)
        return hparams.default_root_dir.joinpath(f'{system_cls.__name__}.ckpt')

    @classmethod
    def _add_system_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds system-specific subparsers/arguments to a parser object.

        The hierarchy of the added arguments can be arbitrarily complex, as long as ``_get_selected_system`` can
        pinpoint a single ``VitalSystem`` to run.

        Args:
            parser: parser object to which system-specific arguments will be added.

        Returns:
            parser object to which system-specific arguments have been added.
        """
        raise NotImplementedError

    @classmethod
    def _override_trainer_default(cls, parser: ArgumentParser) -> ArgumentParser:
        """Allows for overriding generic default trainer attributes with trainer-specific defaults.

        Args:
            parser: parser object that already possesses trainer attributes.

        Returns:
            parser object with overridden default trainer attributes.
        """
        return parser

    @classmethod
    def _get_selected_system(cls, hparams: Namespace) -> Type[VitalSystem]:
        """Identify, through the parameters specified in the CLI, the type of the Lightning module chosen by the user.

        Args:
            hparams: arguments parsed from the CLI.

        Returns:
            type of the Lightning module selected by the user to be run.
        """
        raise NotImplementedError

    @classmethod
    def _add_generic_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds generic custom arguments for running a system to a parser object.

        Args:
            parser: parser object to which generic custom arguments will be added.

        Returns
            parser object to which generic custom arguments have been added.
        """
        # save/load parameters
        parser.add_argument("--pretrained", type=Path, help="Path to Lightning module checkpoints to restore system")

        # evaluation parameters
        parser.add_argument("--predict", action='store_true', help="Skip training and do test phase")

        # resource-use parameters
        parser.add_argument('--workers', type=int, default=os.cpu_count() - 1,
                            help="How many subprocesses to use for data loading. "
                                 "0 means that the data will be loaded in the main process.")

        return parser

    @classmethod
    def _parse_and_check_args(cls, parser: ArgumentParser) -> Namespace:
        """Parse args, making custom checks on the values of the parameters in the process.

        Args:
            parser: complete parser object for which to make custom checks on the values of the parameters.

        Returns:
            parsed and validated arguments to a system.
        """
        args = parser.parse_args()

        # If output dir is specified, cast it ot PurePath
        if args.default_root_dir is not None:
            args.default_root_dir = Path(args.default_root_dir)

        return args
