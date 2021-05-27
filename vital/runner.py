from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Type

import hydra
import dotenv
from config.system.modules.unet import UNetConfig

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from vital.data.data_module import VitalDataModule
from vital.systems.system import VitalSystem
from vital.utils.config import read_ini_config
from vital.utils.logging import configure_logging

from config.conf import DefaultConfig
from config.data.acdc import AcdcConfig
from config.data.camus import CamusConfig
from config.data.mnist import MnistConfig
from config.system.modules.mlp import MLPConfig
from config.system.classification import ClassificationConfig
from config.system.segmentation import SegmentationConfig


class VitalRunner(ABC):
    """Abstract runner that runs the main training/val loop, etc. using Lightning Trainer."""

    @classmethod
    def create_configs(cls, cs: ConfigStore):
        cs.store(name="default", node=DefaultConfig)

    @classmethod
    def store_groups(cls, cs: ConfigStore):
        configuration = {
            "data": {"acdc": AcdcConfig, "camus": CamusConfig, "mnist": MnistConfig},
            "system": {'segmentation': SegmentationConfig, 'classification': ClassificationConfig},
            # 'system.module': {'mlp': MLPConfig, 'unet': UNetConfig}
            'module': {'mlp': MLPConfig, 'unet': UNetConfig}
        }

        for group_name, group in configuration.items():
            for name, node in group.items():
                cs.store(group=group_name, name=name, node=node)

    @classmethod
    def main(cls) -> None:

        # load environment variables from `.env` file if it exists
        dotenv.load_dotenv(override=True)

        cs = ConfigStore.instance()
        cls.create_configs(cs)
        cls.store_groups(cs)
        # cls.run_system()

    @classmethod
    def run_system(cls, cfg: DictConfig) -> None:
        """Handles the training and evaluation of a model.

        Args:
            cfg:
        """

        seed_everything(cfg.seed)

        # # Use Comet for logging if a path to a Comet config file is provided
        # # and logging is enabled in Lightning (i.e. `fast_dev_run=False`)
        logger = True
        # if hparams.comet_config and not hparams.fast_dev_run:
        #     logger = cls._configure_comet_logger(hparams)

        if cfg.resume:
            trainer = Trainer(resume_from_checkpoint=cfg.ckpt_path, logger=logger)
        else:
            trainer = hydra.utils.instantiate(cfg.trainer)

        # If logger as a logger directory, use it. Otherwise, default to using `default_root_dir`
        log_dir = Path(trainer.log_dir) if trainer.log_dir else cfg.trainer.default_root_dir

        if not cfg.trainer.fast_dev_run:
            # Configure Python logging right after instantiating the trainer (which determines the logs' path)
            cls._configure_logging(log_dir, None)

        datamodule = hydra.utils.instantiate(cfg.data)

        module = hydra.utils.instantiate(cfg.module,
                                         input_shape=datamodule.data_params.in_shape,
                                         ouput_shape=datamodule.data_params.out_shape)

        print(cfg.system)
        model = hydra.utils.instantiate(cfg.system, module, datamodule.data_params)

        if cfg.ckpt_path:  # and not hparams.weights_only:  # Load pretrained model if checkpoint is provided
            model = model.load_from_checkpoint(str(cfg.ckpt_path), **cfg.system)
        # else:
        #     model = system_cls(**vars(hparams), data_params=datamodule.data_params)
        #     if hparams.ckpt_path and hparams.weights_only:
        #         checkpoint = torch.load(hparams.weights, map_location=model.device)
        #         model.load_state_dict(checkpoint["state_dict"], strict=hparams.strict_load)

        if cfg.train:
            trainer.fit(model, datamodule=datamodule)

            if not cfg.trainer.fast_dev_run:
                # Copy best model checkpoint to a predictable path + online tracker (if used)
                # best_model_path = cls._best_model_path(log_dir, hparams)
                # copy2(trainer.checkpoint_callback.best_model_path, str(best_model_path))

                # if hparams.comet_config:
                #     trainer.logger.experiment.log_model("model", trainer.checkpoint_callback.best_model_path)

                # Ensure we use the best weights (and not the latest ones) by loading back the best model
                model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        if cfg.test:
            trainer.test(model, datamodule=datamodule)

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
        offline_kwargs = {"offline": comet_config.getboolean("offline", fallback=False)}
        if "offline" in comet_config:
            del comet_config["offline"]
            offline_kwargs["save_dir"] = str(hparams.default_root_dir)
        return CometLogger(**dict(comet_config), **offline_kwargs)

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
    def _get_selected_data_module(cls, hparams: Namespace) -> Type[VitalDataModule]:
        """Identify, through the parameters specified in the CLI, the type of data module chosen by the user.

        Args:
            hparams: Arguments parsed from the CLI.

        Returns:
            Type of the data module selected by the user to be provided to the Lightning module.
        """
        raise NotImplementedError

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

        # save/load parameters
        parser.add_argument("--ckpt_path", type=Path, help="Path to Lightning module checkpoints to restore system")
        parser.add_argument("--weights_only", action="store_true", help="Load only weights from ckpt_path")
        parser.add_argument(
            "--no_strict_load",
            dest="strict_load",
            action="store_false",
            help="Disable strict enforcing of keys when loading state dict",
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
        parser.add_argument("--seed", type=int, help="Seed for reproducibility. If None, seed will be set randomly")

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


if __name__ == "__main__":
    VitalRunner.main()


    @hydra.main(config_name="default")
    def run(cfg: DictConfig):
        VitalRunner.run_system(cfg)


    run()
