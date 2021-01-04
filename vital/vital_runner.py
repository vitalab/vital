import os
from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import copy2
from typing import List, Type

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from vital.systems.vital_system import VitalSystem
from vital.utils.logging import configure_logging
from vital.utils.parsing import StoreDictKeyPair


class VitalRunner(ABC):
    """Abstract runner that runs the main training/val loop, etc... using Lightning Trainer."""

    @classmethod
    def main(cls) -> None:
        """Sets-up the CLI for the ``LightningModule``s runnable through this trainer and runs the requested system."""
        # Initialize the parser with our own generic arguments
        parser = ArgumentParser()
        parser = cls._add_generic_args(parser)

        # Add Lightning trainer arguments to the parser
        parser = Trainer.add_argparse_args(parser)

        # Override Lightning trainer defaults with trainer-specific defaults
        parser = cls._override_trainer_default(parser)

        # Add subparsers for all systems available through the trainer
        parser = cls._add_system_args(parser)

        # Parse args
        hparams = cls._parse_and_check_args(parser)

        # Configure logging right after args parsing,
        # since this allows customization of logging behavior
        cls._configure_logging(hparams)

        # Run target system
        cls.run_system(hparams)

    @classmethod
    def run_system(cls, hparams: Namespace) -> None:
        """Handles the training and evaluation of a model.

        Args:
            hparams: Arguments parsed from the CLI.
        """
        system_cls = cls._get_selected_system(hparams)

        if hparams.resume:
            trainer = Trainer(resume_from_checkpoint=hparams.ckpt_path)
        else:
            trainer = Trainer.from_argparse_args(
                hparams,
                callbacks=[
                    ModelCheckpoint(**hparams.model_checkpoint_kwargs),
                    EarlyStopping(**hparams.early_stopping_kwargs),
                    *cls._get_callbacks(hparams),
                ],
            )

        if hparams.ckpt_path:  # Load pretrained model if checkpoint is provided
            model = system_cls.load_from_checkpoint(str(hparams.ckpt_path), **vars(hparams))
        else:
            model = system_cls(**vars(hparams))

        if hparams.train:
            trainer.fit(model)

            # Copy best model checkpoint to a fixed path
            best_model_path = cls._define_best_model_save_path(hparams)
            copy2(str(trainer.checkpoint_callback.best_model_path), str(best_model_path))

            # Ensure we use the best weights (and not the latest ones) by loading back the best model
            model = system_cls.load_from_checkpoint(str(best_model_path))

        if hparams.test:
            trainer.test(model)

    @classmethod
    def _configure_logging(cls, hparams: Namespace) -> None:
        """Callback that defines the default logging behavior.

        It can be overridden to customize the logging behavior, e.g. to adjust to some CLI arguments defined by the
        user.

        Args:
            hparams: Arguments parsed from the CLI.
        """
        configure_logging(log_to_console=True, log_file=hparams.default_root_dir / "run.log")

    @classmethod
    def _define_best_model_save_path(cls, hparams: Namespace) -> Path:
        """Defines the fixed path (w.r.t to the system to run) where to copy the best model checkpoint after training.

        Args:
            hparams: Arguments parsed from the CLI.

        Returns:
            Fixed path (w.r.t to the system to run) where to copy the best model checkpoint after training.
        """
        system_cls = cls._get_selected_system(hparams)
        return hparams.default_root_dir / f"{system_cls.__name__}.ckpt"

    @classmethod
    def _add_system_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds system-specific subparsers/arguments to a parser object.

        The hierarchy of the added arguments can be arbitrarily complex, as long as ``_get_selected_system`` can
        pinpoint a single ``VitalSystem`` to run.

        Args:
            parser: Parser object to which system-specific arguments will be added.

        Returns:
            Parser object to which system-specific arguments have been added.
        """
        raise NotImplementedError

    @classmethod
    def _override_trainer_default(cls, parser: ArgumentParser) -> ArgumentParser:
        """Allows for overriding Lightning trainer default attributes with runner-specific defaults.

        Args:
            parser: Parser object that already possesses trainer attributes.

        Returns:
            Parser object with overridden trainer attributes.
        """
        return parser

    @classmethod
    def _get_selected_system(cls, hparams: Namespace) -> Type[VitalSystem]:
        """Identify, through the parameters specified in the CLI, the type of the Lightning module chosen by the user.

        Args:
            hparams: Arguments parsed from the CLI.

        Returns:
            Type of the Lightning module selected by the user to be run.
        """
        raise NotImplementedError

    @classmethod
    def _get_callbacks(cls, hparams: Namespace) -> List[Callback]:
        """Initialize, through the parameters specified in the CLI, the callbacks to use in this run.

        Args:
            hparams: Arguments parsed from the CLI.

        Returns:
            Callbacks to pass to the Lightning `Trainer`.
        """
        return []

    @classmethod
    def _add_generic_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds generic custom arguments for running a system to a parser object.

        Args:
            parser: Parser object to which generic custom arguments will be added.

        Returns:
            Parser object to which generic custom arguments have been added.
        """
        # callback parameters
        parser.add_argument(
            "--model_checkpoint_kwargs",
            type=StoreDictKeyPair,
            default=dict(),
            help="Parameters for Lightning's built-in model checkpoint callback",
        )
        parser.add_argument(
            "--early_stopping_kwargs",
            type=StoreDictKeyPair,
            default=dict(),
            help="Parameters for Lightning's built-in early stopping callback",
        )

        # save/load parameters
        parser.add_argument("--ckpt_path", type=Path, help="Path to Lightning module checkpoints to restore system")
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Disregard any other CLI configuration and restore exact state from the checkpoint",
        )

        # run parameters
        parser.add_argument(
            "--skip_train", dest="train", action="store_false", help="Skip training and do test/evaluation phase"
        )
        parser.add_argument("--skip_test", dest="test", action="store_false", help="Skip test/evaluation phase")

        # resource-use parameters
        parser.add_argument(
            "--num_workers",
            type=int,
            default=os.cpu_count() - 1,
            help="How many subprocesses to use for data loading. "
            "0 means that the data will be loaded in the main process.",
        )

        return parser

    @classmethod
    def _parse_and_check_args(cls, parser: ArgumentParser) -> Namespace:
        """Parse args, making custom checks on the values of the parameters in the process.

        Args:
            parser: Complete parser object for which to make custom checks on the values of the parameters.

        Returns:
            Parsed and validated arguments for a system run.

        Raises:
            ValueError: If invalid combinations of arguments are specified by the user.
                - ``--skip_train`` flag is active without a ``--checkpoint`` being provided.
                - ``--resume`` flag is active without a ``--checkpoint`` being provided.
        """
        args = parser.parse_args()

        if not args.ckpt_path:
            if not args.train:
                raise ValueError(
                    "Trainer set to skip training (`--skip_train` flag) without a checkpoint provided. \n"
                    "Either allow model to train (remove `--skip_train` flag) or "
                    "provide a pretrained model (through `--ckpt_path` parameter)."
                )
            if args.resume:
                raise ValueError(
                    "Cannot use flag `--resume` without a checkpoint from which to resume. \n"
                    "Either allow the model to start over (remove `--resume` flag) or "
                    "provide a saved checkpoint (through `--ckpt_path` flag)"
                )

        if args.default_root_dir is None:
            # If no output dir is specified, default to the working directory
            args.default_root_dir = Path.cwd()
        else:
            # If output dir is specified, cast it os Path
            args.default_root_dir = Path(args.default_root_dir)

        return args
