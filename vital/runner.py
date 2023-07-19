import logging
import os
from abc import ABC
from argparse import Namespace
from pathlib import Path
from shutil import copy2
from typing import Union

import comet_ml  # noqa
import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, Logger

from vital.data.data_module import VitalDataModule
from vital.system import VitalSystem
from vital.utils.config import (
    instantiate_config_node_leaves,
    instantiate_results_processor,
    register_omegaconf_resolvers,
)
from vital.utils.saving import resolve_model_checkpoint_path

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
        load_dotenv()

        register_omegaconf_resolvers()

    @staticmethod
    @hydra.main(version_base=None, config_path="config", config_name="vital_default")
    def run_system(cfg: DictConfig) -> None:
        """Handles the training and evaluation of a model.

        Note: Must be static because of the hydra.main decorator and config pass-through.

        Args:
            cfg: Configuration to run the experiment.
        """
        cfg = VitalRunner._check_cfg(cfg)

        # Global torch config making it possible to use performant matrix multiplications on Ampere and newer CUDA GPUs
        # `vital` default value of is `high`, the middle-ground between performance and precision. For more details:
        # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

        if cfg.ckpt:
            ckpt_path = resolve_model_checkpoint_path(cfg.ckpt)

        cfg.seed = seed_everything(cfg.seed, workers=True)
        experiment_logger = VitalRunner.configure_logger(cfg)

        # Instantiate post-processing objects
        postprocessors = []
        if isinstance(postprocessing_node := cfg.data.get("postprocessing"), DictConfig):
            postprocessors = instantiate_config_node_leaves(postprocessing_node, "post-processing")

        # Instantiate the different types of callbacks from the configs
        callbacks = []
        if isinstance(callbacks_node := cfg.get("callbacks"), DictConfig):
            callbacks.extend(instantiate_config_node_leaves(callbacks_node, "callback"))
        if isinstance(results_processors_node := cfg.get("results_processors"), DictConfig):
            callbacks.extend(
                instantiate_config_node_leaves(
                    results_processors_node, "results processor", instantiate_fn=instantiate_results_processor
                )
            )
        if isinstance(predict_node := cfg.data.get("predict"), DictConfig):
            logger.info("Instantiating prediction writer")
            prediction_writer_kwargs = {}
            if postprocessors:
                prediction_writer_kwargs["postprocessors"] = postprocessors
            callbacks.append(hydra.utils.instantiate(predict_node, **prediction_writer_kwargs))

        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=experiment_logger, callbacks=callbacks)
        trainer.logger.log_hyperparams(Namespace(**cfg))  # Save config to logger.

        if isinstance(trainer.logger, CometLogger):
            experiment_logger.experiment.log_asset_folder(".hydra", log_file_name=True)
            if cfg.get("comet_tags", None):
                experiment_logger.experiment.add_tags(list(cfg.comet_tags))

        # Instantiate datamodule
        datamodule: VitalDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)

        # Instantiate system (which will handle instantiating the model and optimizer).
        model: VitalSystem = hydra.utils.instantiate(
            cfg.task, choices=cfg.choices, data_params=datamodule.data_params, _recursive_=False
        )

        if cfg.ckpt:  # Load pretrained model if checkpoint is provided
            if cfg.weights_only:
                logger.info(f"Loading weights from {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path, map_location=model.device)["state_dict"], strict=cfg.strict)
            else:
                logger.info(f"Loading model from {ckpt_path}")
                model = model.load_from_checkpoint(ckpt_path, data_params=datamodule.data_params, strict=cfg.strict)

        if cfg.train:
            if cfg.resume:
                trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
            else:
                trainer.fit(model, datamodule=datamodule)

            if not cfg.trainer.get("fast_dev_run", False):
                # Copy best model checkpoint to a predictable path + online tracker (if used)
                best_model_path = VitalRunner._best_model_path(model.log_dir, cfg)
                if trainer.checkpoint_callback is not None:
                    copy2(trainer.checkpoint_callback.best_model_path, str(best_model_path))
                    # Ensure we use the best weights (and not the latest ones) by loading back the best model
                    model = model.load_from_checkpoint(str(best_model_path))
                else:  # If checkpoint callback is not used, save current model.
                    trainer.save_checkpoint(best_model_path)

                if isinstance(trainer.logger, CometLogger):
                    last_model_path = None
                    if trainer.checkpoint_callback is not None:
                        best_model_path = trainer.checkpoint_callback.best_model_path
                        last_model_path = trainer.checkpoint_callback.last_model_path

                    trainer.logger.experiment.log_model("best-model", best_model_path)

                    # Also log the `ModelCheckpoint`'s last checkpoint, if it is configured to save one
                    if last_model_path:
                        trainer.logger.experiment.log_model("last-model", last_model_path)

        if cfg.test:
            trainer.test(model, datamodule=datamodule)

        if cfg.predict:
            trainer.predict(model, datamodule=datamodule)

    @staticmethod
    def _check_cfg(cfg: DictConfig) -> DictConfig:
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
    def configure_logger(cfg: DictConfig) -> Union[bool, Logger]:
        """Initializes Lightning logger.

        Args:
            cfg: Full configuration for the experiment.

        Returns:
            Logger for the Lightning Trainer.
        """
        experiment_logger = True  # Default to True (Tensorboard)
        skip_logger = False
        # Configure custom logger only if user specified custom config
        if "logger" in cfg and isinstance(cfg.logger, DictConfig):
            if "_target_" not in cfg.logger:
                logger.warning("No _target_ in logger config. Cannot instantiate custom logger")
                skip_logger = True
            if cfg.trainer.get("fast_dev_run", False):
                logger.warning(
                    "Not instantiating custom logger because having `fast_dev_run=True` makes Lightning skip logging. "
                    "To test the logger, launch a full run."
                )
                skip_logger = True
            if not skip_logger and "_target_" in cfg.logger:
                if "comet" in cfg.logger._target_:
                    experiment_logger = hydra.utils.instantiate(cfg.logger)
                elif "tensorboard" in cfg.logger._target_:
                    # If no save_dir is passed, use default logger and let Trainer set save_dir.
                    if cfg.logger.get("save_dir", None):
                        experiment_logger = hydra.utils.instantiate(cfg.logger)
        return experiment_logger

    @staticmethod
    def _best_model_path(log_dir: Path, cfg: DictConfig) -> Path:
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
            model = cfg.choices["task/model"]
            name = f"{cfg.choices.data}_{cfg.choices.task}"
            if model is not None:  # Some systems do not have a model (ex. Auto-encoders)
                name = f"{name}_{model}"
            return log_dir / f"{name}.ckpt"


def main():
    """Run the script."""
    VitalRunner.main()


if __name__ == "__main__":
    main()
