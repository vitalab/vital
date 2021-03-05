from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import copy2
from typing import List, Type

import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from vital.systems.system import VitalSystem
from vital.utils.config import read_ini_config
from vital.utils.logging import configure_logging
from vital.utils.parsing import StoreDictKeyPair


class VitalRunner(ABC):
    """Abstract runner that runs the main training/val loop, etc. using Lightning Trainer."""

    @classmethod
    def main(cls) -> None:
        """Sets-up the CLI for the ``LightningModule``s runnable through this trainer and runs the requested system."""
        # Initialize the parser with our own generic arguments, Lightning trainer arguments,
        # and subparsers for all systems available through the trainer
        parser = cls._add_system_args(
            cls._override_trainer_default(cls._add_generic_args(Trainer.add_argparse_args(ArgumentParser())))
        )

        # Run target system
        cls.run_system(cls._parse_and_check_args(parser))

    @classmethod
    def run_system(cls, hparams: Namespace) -> None:
        """Handles the training and evaluation of a model.

        Args:
            hparams: Arguments parsed from the CLI.
        """
        seed_everything(hparams.seed)

        # Use Comet for logging if a path to a Comet config file is provided
        # and logging is enabled in Lightning (i.e. `fast_dev_run=False`)
        logger = True
        if hparams.comet_config and not hparams.fast_dev_run:
            logger = cls._configure_comet_logger(hparams)

        if hparams.resume:
            trainer = Trainer(resume_from_checkpoint=hparams.ckpt_path, logger=logger)
        else:
            trainer = Trainer.from_argparse_args(
                hparams,
                callbacks=[
                    ModelCheckpoint(**hparams.model_checkpoint_kwargs),
                    EarlyStopping(**hparams.early_stopping_kwargs),
                    *cls._get_callbacks(hparams),
                ],
                logger=logger,
            )

        # If logger as a logger directory, use it. Otherwise, default to using `default_root_dir`
        log_dir = Path(trainer.log_dir) if trainer.log_dir else hparams.default_root_dir

        if not hparams.fast_dev_run:
            # Configure Python logging right after instantiating the trainer (which determines the logs' path)
            cls._configure_logging(log_dir, hparams)

        system_cls = cls._get_selected_system(hparams)
        if hparams.ckpt_path:  # Load pretrained model if checkpoint is provided
            model = system_cls.load_from_checkpoint(str(hparams.ckpt_path), **vars(hparams))
        else:
            model = system_cls(**vars(hparams))
            if hparams.weights:
                map_location = None if torch.cuda.is_available() else torch.device("cpu")
                checkpoint = torch.load(hparams.weights, map_location=map_location)
                model.load_state_dict(checkpoint["state_dict"], strict=False)

        if hparams.train:
            trainer.fit(model)

            if not hparams.fast_dev_run:
                # Copy best model checkpoint to a predictable path
                best_model_path = cls._best_model_path(log_dir, hparams)
                copy2(trainer.checkpoint_callback.best_model_path, str(best_model_path))

                # Ensure we use the best weights (and not the latest ones) by loading back the best model
                model = system_cls.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        if hparams.test:
            trainer.test(model)

    @classmethod
    def _configure_logging(cls, log_dir: Path, hparams: Namespace) -> None:
        """Callback that defines the default logging behavior.

        It can be overridden to customize the logging behavior, e.g. to adjust to some CLI arguments defined by the
        user.

        Args:
            log_dir: Lightning's directory for the current run.
            hparams: Arguments parsed from the CLI.
        """
        configure_logging(log_to_console=True, log_file=log_dir / "run.log")

    @classmethod
    def _configure_comet_logger(cls, hparams: Namespace) -> CometLogger:
        """Builds a ``CometLogger`` instance using the content of the Comet configuration file.

        Notes:
            - The Comet configuration file should follow the `.comet.config` format. See Comet's documentation for more
              details: https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables

        Args:
            hparams: Arguments parsed from the CLI.

        Returns:
            Instance of ``CometLogger`` built using the content of the Comet configuration file.
        """
        comet_config = read_ini_config(hparams.comet_config)["comet"]
        offline = comet_config.getboolean("offline", fallback=False)
        if "offline" in comet_config:
            del comet_config["offline"]
        return CometLogger(**dict(comet_config), offline=offline)

    @classmethod
    def _best_model_path(cls, log_dir: Path, hparams: Namespace) -> Path:
        """Defines the path where to copy the best model checkpoint after training.

        Args:
            log_dir: Lightning's directory for the current run.
            hparams: Arguments parsed from the CLI.

        Returns:
            Path where to copy the best model checkpoint after training.
        """
        return log_dir / f"{cls._get_selected_system(hparams).__name__}.ckpt"

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
        """Adds to the parser object some generic arguments useful for running a system.

        Args:
            parser: Parser object to which generic custom arguments will be added.

        Returns:
            Parser object to which generic custom arguments have been added.
        """
        # logging parameters
        parser.add_argument(
            "--comet_config",
            type=Path,
            help="Path to Comet configuration file, if you want to track the experiment using Comet",
        )

        # callback parameters
        parser.add_argument(
            "--model_checkpoint_kwargs",
            action=StoreDictKeyPair,
            default=dict(),
            metavar="ARG1=VAL1,ARG2=VAL2...",
            help="Parameters for Lightning's built-in model checkpoint callback",
        )
        parser.add_argument(
            "--early_stopping_kwargs",
            action=StoreDictKeyPair,
            default=dict(),
            metavar="ARG1=VAL1,ARG2=VAL2...",
            help="Parameters for Lightning's built-in early stopping callback",
        )

        # save/load parameters
        loading_group = parser.add_mutually_exclusive_group()
        loading_group.add_argument(
            "--ckpt_path", type=Path, help="Path to Lightning module checkpoints to restore system"
        )
        loading_group.add_argument(
            "--weights", type=Path, help="Path to Lightning module checkpoints to restore system weights"
        )
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

        # seed parameter
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Seed for reproducibility. If None, seed will be set randomly",
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
