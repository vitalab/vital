import os
from abc import ABC
from argparse import Namespace
from pathlib import Path
from shutil import copy2
from typing import List

import dotenv
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import CometLogger
from vital.data.data_module import VitalDataModule
from vital.systems.system import VitalSystem
from vital.utils.config import read_ini_config
from vital.utils.logging import configure_logging


class VitalRunner(ABC):
    """Abstract runner that runs the main training/val loop, etc. using Lightning Trainer."""

    @classmethod
    def main(cls) -> None:
        # Load environment variables from `.env` file if it exists
        # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
        dotenv.load_dotenv(override=True)

        cls.run_system()

    @staticmethod
    @hydra.main(config_path='config_example', config_name='default.yaml')
    def run_system(cfg: DictConfig) -> None:
        """Handles the training and evaluation of a model.

        Note: Must be static because of the hydra.main decorator and config pass-through.

        Args:
            cfg: Configuration to run the experiment.
        """
        cfg = VitalRunner._check_cfg(cfg)

        seed_everything(cfg.seed)

        # Use Comet for logging if a path to a Comet config file is provided
        # and logging is enabled in Lightning (i.e. `fast_dev_run=False`)
        logger = True
        # if hparams.comet_config and not hparams.fast_dev_run:
        #     logger = cls._configure_comet_logger(hparams)

        # Init Lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in cfg:
            for conf_name, conf in cfg.callbacks.items():
                print(f"Instantiating callback <{conf_name}>")
                callbacks.append(hydra.utils.instantiate(conf))

        if cfg.resume:
            trainer = Trainer(resume_from_checkpoint=cfg.ckpt_path, logger=logger, callbacks=callbacks)
        else:
            trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

        # If logger as a logger directory, use it. Otherwise, default to using `default_root_dir`
        log_dir = Path(trainer.log_dir) if trainer.log_dir else cfg.trainer.default_root_dir

        if not cfg.trainer.get('fast_dev_run', False):
            # Configure Python logging right after instantiating the trainer (which determines the logs' path)
            VitalRunner._configure_logging(log_dir, cfg)

        datamodule: VitalDataModule = hydra.utils.instantiate(cfg.data)

        module: nn.Module = hydra.utils.instantiate(cfg.module,
                                                    input_shape=datamodule.data_params.in_shape,
                                                    ouput_shape=datamodule.data_params.out_shape)

        model: VitalSystem = hydra.utils.instantiate(cfg.system, module, datamodule.data_params)

        if cfg.ckpt_path and not cfg.weights_only:  # Load pretrained model if checkpoint is provided
            if cfg.weights_only:
                checkpoint = torch.load(cfg.ckpt_path, map_location=model.device)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model = model.load_from_checkpoint(str(cfg.ckpt_path), **cfg.system)

        if cfg.train:
            trainer.fit(model, datamodule=datamodule)

            if not cfg.trainer.get('fast_dev_run', False):
                # # Copy best model checkpoint to a predictable path + online tracker (if used)
                best_model_path = VitalRunner._best_model_path(log_dir, cfg)
                print(best_model_path)
                copy2(trainer.checkpoint_callback.best_model_path, str(best_model_path))

                # if hparams.comet_config:
                #     trainer.logger.experiment.log_model("model", trainer.checkpoint_callback.best_model_path)

                # Ensure we use the best weights (and not the latest ones) by loading back the best model
                # TODO fix TypeError: __init__() missing 1 required positional argument: 'module'
                # print(torch.load(trainer.checkpoint_callback.best_model_path, map_location=model.device))
                # model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
                model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"])

        if cfg.test:
            trainer.test(model, datamodule=datamodule)

    @classmethod
    def _configure_logging(cls, log_dir: Path, cfg: DictConfig) -> None:
        """Callback that defines the default logging behavior.

        It can be overridden to customize the logging behavior, e.g. to adjust to some CLI arguments defined by the
        user.

        Args:
            log_dir: Lightning's directory for the current run.
            cfg:
        """
        configure_logging(log_to_console=True, log_file=log_dir / "run.log")

    # @classmethod
    # def _configure_comet_logger(cls, hparams: Namespace) -> CometLogger:
    #     """Builds a ``CometLogger`` instance using the content of the Comet configuration file.
    #
    #     Notes:
    #         - The Comet configuration file should follow the `.comet.config` format. See Comet's documentation for more
    #           details: https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables
    #
    #     Args:
    #         hparams: Arguments parsed from the CLI.
    #
    #     Returns:
    #         Instance of ``CometLogger`` built using the content of the Comet configuration file.
    #     """
    #     comet_config = read_ini_config(hparams.comet_config)["comet"]
    #     offline_kwargs = {"offline": comet_config.getboolean("offline", fallback=False)}
    #     if "offline" in comet_config:
    #         del comet_config["offline"]
    #         offline_kwargs["save_dir"] = str(hparams.default_root_dir)
    #     return CometLogger(**dict(comet_config), **offline_kwargs)

    @classmethod
    def _best_model_path(cls, log_dir: Path, cfg: DictConfig) -> Path:
        """Defines the path where to copy the best model checkpoint after training.

        Args:
            log_dir: Lightning's directory for the current run.
            cfg:

        Returns:
            Path where to copy the best model checkpoint after training.
        """
        data = cfg.data._target_.split('.')[-1]
        system = cfg.system._target_.split('.')[-1]
        module = cfg.module._target_.split('.')[-1]
        return log_dir / f"{data}_{system}_{module}.ckpt"

    @classmethod
    def _check_cfg(cls, cfg: DictConfig):
        """Parse args, making custom checks on the values of the parameters in the process.

        Args:
            cfg:

        Returns:
             Validated config for a system run.

        Raises:
            ValueError: If invalid combinations of arguments are specified by the user.
                - ``train=False`` flag is active without a ``ckpt_path`` being provided.
                - ``resume=True`` flag is active without a ``ckpt_path`` being provided.
        """

        if not cfg.ckpt_path:
            if not cfg.train:
                raise ValueError(
                    "Trainer set to skip training (`train=False` flag) without a checkpoint provided. \n"
                    "Either allow model to train (`train=True` flag) or "
                    "provide a pretrained model (through `ckpt_path=<something>` parameter)."
                )
            if cfg.resume:
                raise ValueError(
                    "Cannot use flag `resume=True` without a checkpoint from which to resume. \n"
                    "Either allow the model to start over (`resume=False` flag) or "
                    "provide a saved checkpoint (through `ckpt_path=<something>` parameter)"
                )

        if cfg.trainer.default_root_dir is None:
            # If no output dir is specified, default to the working directory
            cfg.trainer.default_root_dir = os.getcwd()
            # cfg.trainer.default_root_dir = Path.cwd()
        # else:
        #     # If output dir is specified, cast it os Path
        #     cfg.trainer.default_root_dir = Path(cfg.trainer.default_root_dir)

        if 'gpus' not in cfg.trainer.keys():
            with open_dict(cfg):
                cfg.trainer.gpus = int(torch.cuda.is_available())

        return cfg


if __name__ == "__main__":
    VitalRunner.main()