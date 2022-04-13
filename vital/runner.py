import logging
import os
from abc import ABC
from argparse import Namespace
from pathlib import Path
from shutil import copy2
from typing import List, Optional, Union

import comet_ml  # noqa
import dotenv
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, LightningLoggerBase

from vital.data.data_module import VitalDataModule
from vital.systems.system import VitalSystem
from vital.utils.serialization import resolve_model_checkpoint_path

logger = logging.getLogger(__name__)


class VitalRunner(ABC):
    """Abstract runner that runs the main training/val loop, etc. using Lightning Trainer."""

    @classmethod
    def main(cls) -> None:
        """Runs the requested experiment."""
        # Set up the environment
        cls.pre_run_routine()

        # Run the system with config loaded by @hydra.main
        cls.run_system()

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        # Load environment variables from `.env` file if it exists
        # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
        dotenv.load_dotenv(override=True)

        OmegaConf.register_new_resolver("sys.gpus", lambda x=None: int(torch.cuda.is_available()))
        OmegaConf.register_new_resolver("sys.num_workers", lambda x=None: os.cpu_count() - 1)

    @staticmethod
    @hydra.main(config_path="config_example", config_name="default.yaml")
    def run_system(cfg: DictConfig) -> None:
        """Handles the training and evaluation of a model.

        Note: Must be static because of the hydra.main decorator and config pass-through.

        Args:
            cfg: Configuration to run the experiment.
        """
        cfg = VitalRunner._check_cfg(cfg)

        if cfg.ckpt:
            ckpt_path = resolve_model_checkpoint_path(cfg.ckpt)

        cfg.seed = seed_everything(cfg.seed, workers=True)

        callbacks = VitalRunner.configure_callbacks(cfg)
        experiment_logger = VitalRunner.configure_logger(cfg)

        if cfg.resume:
            trainer = Trainer(resume_from_checkpoint=cfg.ckpt_path, logger=experiment_logger, callbacks=callbacks)
        else:
            trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=experiment_logger, callbacks=callbacks)

            trainer.logger.log_hyperparams(Namespace(**cfg))  # Save config to logger.

        if isinstance(trainer.logger, CometLogger):
            experiment_logger.experiment.log_asset_folder(".hydra", log_file_name=True)
            if cfg.get("comet_tags", None):
                experiment_logger.experiment.add_tags(list(cfg.comet_tags))

        # Instantiate datamodule
        datamodule: VitalDataModule = hydra.utils.instantiate(cfg.data)

        # Instantiate module with respect to datamodule's data params.
        module: nn.Module = hydra.utils.instantiate(
            cfg.system.module,
            input_shape=datamodule.data_params.in_shape,
            output_shape=datamodule.data_params.out_shape,
        )

        # Instantiate model with the created module.
        model: VitalSystem = hydra.utils.instantiate(
            cfg.system, module=module, data_params=datamodule.data_params, _recursive_=False
        )

        if cfg.ckpt:  # Load pretrained model if checkpoint is provided
            if cfg.weights_only:
                logger.info(f"Loading weights from {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path, map_location=model.device)["state_dict"], strict=cfg.strict)
            else:
                logger.info(f"Loading model from {ckpt_path}")
                model = model.load_from_checkpoint(
                    ckpt_path, module=module, data_params=datamodule.data_params, strict=cfg.strict
                )

        if cfg.train:
            trainer.fit(model, datamodule=datamodule)

            if not cfg.trainer.get("fast_dev_run", False):
                # Copy best model checkpoint to a predictable path + online tracker (if used)
                best_model_path = VitalRunner._best_model_path(model.log_dir, cfg)
                if trainer.checkpoint_callback is not None:
                    copy2(trainer.checkpoint_callback.best_model_path, str(best_model_path))
                    # Ensure we use the best weights (and not the latest ones) by loading back the best model
                    model = model.load_from_checkpoint(str(best_model_path), module=module)
                else:  # If checkpoint callback is not used, save current model.
                    trainer.save_checkpoint(best_model_path)

                if isinstance(trainer.logger, CometLogger):
                    trainer.logger.experiment.log_model("model", trainer.checkpoint_callback.best_model_path)

        if cfg.test:
            trainer.test(model, datamodule=datamodule)

    @classmethod
    def _check_cfg(cls, cfg: DictConfig) -> DictConfig:
        """Parse args, making custom checks on the values of the parameters in the process.

        Args:
            cfg: Full configuration for the experiment.

        Returns:
             Validated config for a system run.
        """
        # If no output dir is specified, default to the working directory
        if not cfg.trainer.get("default_root_dir", None):
            with open_dict(cfg):
                cfg.trainer.default_root_dir = os.getcwd()

        return cfg

    @staticmethod
    def configure_callbacks(cfg: DictConfig) -> Optional[List[Callback]]:
        """Initializes Lightning callbacks.

        Args:
            cfg: Full configuration for the experiment.

        Returns:
            callbacks for the Lightning Trainer
        """
        if "callbacks" in cfg:
            callbacks = []
            for conf_name, conf in cfg.callbacks.items():
                if "_target_" in conf:
                    logger.info(f"Instantiating callback <{conf_name}>")
                    callbacks.append(hydra.utils.instantiate(conf))
                else:
                    logger.warning(f"No _target_ in callback config. Cannot instantiate {conf_name}")
        else:
            callbacks = None

        return callbacks

    @staticmethod
    def configure_logger(cfg: DictConfig) -> Union[bool, LightningLoggerBase]:
        """Initializes Lightning logger.

        Args:
            cfg: Full configuration for the experiment.

        Returns:
            logger for the Lightning Trainer
        """
        experiment_logger = True  # Default to True (Tensorboard)
        if isinstance(cfg.logger, DictConfig):
            if "_target_" in cfg.logger:
                if "comet" in cfg.logger._target_ and not cfg.trainer.get("fast_dev_run", False):
                    experiment_logger = hydra.utils.instantiate(cfg.logger)
                elif "tensorboard" in cfg.logger._target_:
                    # If no save_dir is passed, use default logger and let Trainer set save_dir.
                    if cfg.logger.get("save_dir", None):
                        experiment_logger = hydra.utils.instantiate(cfg.logger)
            else:
                logger.warning("No _target_ in logger config. Cannot instantiate Logger")
        return experiment_logger

    @classmethod
    def _best_model_path(cls, log_dir: Path, cfg: DictConfig) -> Path:
        """Defines the path where to copy the best model checkpoint after training.

        Args:
            log_dir: Lightning's directory for the current run.
            cfg: Full configuration for the experiment.

        Returns:
            Path where to copy the best model checkpoint after training.
        """
        if cfg.get("best_model_save_path", None):
            return Path(cfg.best_model_save_path)  # Return save path from config if available
        else:
            module = cfg.choices["system/module"]
            name = f"{cfg.choices.data}_{cfg.choices.system}"
            if module is not None:  # Some systems do not have a module (ex. Auto-encoders)
                name = f"{name}_{module}"
            return log_dir / f"{name}.ckpt"


if __name__ == "__main__":
    VitalRunner.main()
