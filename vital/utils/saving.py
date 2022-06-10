import importlib
import logging
import shutil
from pathlib import Path
from typing import Type, Union

import comet_ml
import torch
from packaging.version import InvalidVersion, Version
from pytorch_lightning.core.saving import ModelIO

from vital import get_vital_home
from vital.system import VitalSystem

logger = logging.getLogger(__name__)


def resolve_model_checkpoint_path(checkpoint: Union[str, Path]) -> Path:
    """Resolves a local path or a Comet model registry query to a local path on the machine.

    Notes:
        - If the `ckpt` is to be downloaded off of a Comet model registry, your Comet API key needs to be set in one of
          Comet's expected locations: https://www.comet.ml/docs/python-sdk/advanced/#non-interactive-setup

    Args:
        checkpoint: Location of the checkpoint. This can be either a local path, or the fields of a query to a Comet
            model registry. Examples of different queries:
                - For the latest version of the model: 'my_workspace/my_model'
                - Using a specific version/stage: 'my_workspace/my_model/0.1.0' or 'my_workspace/my_model/prod'

    Returns:
        Path to the model's checkpoint file on the local computer. This can either be the `ckpt` already provided, if it
        was already local, or the location where the checkpoint was downloaded, if it pointed to a Comet registry model.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".ckpt":
        local_ckpt_path = checkpoint
    else:
        try:
            comet_api = comet_ml.api.API()
        except ValueError:
            raise RuntimeError(
                f"The format of the checkpoint '{checkpoint}' indicates you want to download a model from a Comet "
                f"model registry, but Comet couldn't find an API key. Either switch to providing a local checkpoint "
                f"path, or set your Comet API key in one of Comet's expected locations."
            )

        # Parse the provided checkpoint path as a query for a Comet model registry
        version_or_stage, version, stage = None, None, None
        if len(checkpoint.parts) == 2:
            workspace, registry_name = checkpoint.parts
        elif len(checkpoint.parts) == 3:
            workspace, registry_name, version_or_stage = checkpoint.parts
            try:
                Version(version_or_stage)  # Will fail if `version_or_stage` cannot be parsed as a version
                version = version_or_stage
            except InvalidVersion:
                stage = version_or_stage
        else:
            raise ValueError(f"Failed to interpret checkpoint '{checkpoint}' as a query for a Comet model registry.")

        # If neither version nor stage were provided, use latest version available
        if not version_or_stage:
            version_or_stage = version = comet_api.get_registry_model_versions(workspace, registry_name)[-1]

        # Determine where to download the checkpoint locally
        cache_dir = get_vital_home()
        model_cached_path = cache_dir / workspace / registry_name / version_or_stage

        # When using stage, delete cached versions and force re-downloading the registry model,
        # because stage tags can be changed
        if stage:
            shutil.rmtree(model_cached_path, ignore_errors=True)

        # Download model if not already cached
        if not model_cached_path.exists():
            comet_api.download_registry_model(
                workspace, registry_name, version=version, stage=stage, output_path=str(model_cached_path)
            )
        else:
            logger.info(
                f"Using cached registry model {registry_name}, version {version} from workspace {workspace} "
                f"located in '{model_cached_path}'."
            )

        # Extract the path of the checkpoint file on the local machine
        ckpt_files = list(model_cached_path.glob("*.ckpt"))
        if len(ckpt_files) != 1:
            raise RuntimeError(
                f"Expected the Comet model to contain a single '*.ckpt' file, but there were {len(ckpt_files)} "
                f"'*.ckpt' file(s): {ckpt_files}. Either edit the content of the Comet model, or use a different model."
            )
        local_ckpt_path = ckpt_files[0]

    return local_ckpt_path


def load_from_checkpoint(
    checkpoint: Union[str, Path],
    train_mode: bool = False,
    device_type: str = None,
    expected_checkpoint_type: Type[VitalSystem] = None,
) -> VitalSystem:
    """Loads a Lightning module checkpoint, casting it to the appropriate type.

    The module's class is automatically determined based on the hyperparameters saved in the checkpoint.

    Args:
        checkpoint: Location of the checkpoint. This can be either a local path, or the fields of a query to a Comet
            model registry. Examples of different queries:
                - For the latest version of the model: 'my_workspace/my_model'
                - Using a specific version/stage: 'my_workspace/my_model/0.1.0' or 'my_workspace/my_model/prod'
        train_mode: Whether the model should be in 'train' mode (`True`) or 'eval' mode (`False`).
        device_type: Device on which to move the Lightning module after it's been loaded. Defaults to using 'cuda' if
            it is available, and 'cpu' otherwise.
        expected_checkpoint_type: Type of model expected to be loaded from the checkpoint. Used to perform a runtime
            check, and raise an error if the expected model type does not match the loaded model.

    Returns:
        Lightning module loaded from the checkpoint, casted to its original type.
    """
    # Resolve the local path of the checkpoint
    ckpt_path = resolve_model_checkpoint_path(checkpoint)

    # Extract which class to load from the hyperparameters saved in the checkpoint
    ckpt_hparams = torch.load(ckpt_path)[ModelIO.CHECKPOINT_HYPER_PARAMS_KEY]
    model_mod, model_cls_name = ckpt_hparams["task"]["_target_"].rsplit(".", 1)
    model_cls = getattr(importlib.import_module(model_mod), model_cls_name)

    # Restore the model from the checkpoint
    model = model_cls.load_from_checkpoint(str(ckpt_path), ckpt=checkpoint)

    # Perform runtime check on the type of the loaded model
    if expected_checkpoint_type and not isinstance(model, expected_checkpoint_type):
        raise RuntimeError(
            f"Type of the model loaded from the checkpoint does not correspond to the model's expected type. "
            f"Type of the model loaded from checkpoint: {type(model)} "
            f"Expected type of the model: {expected_checkpoint_type}"
        )

    # Set the mode of the model according to the caller's requirements
    model.train(mode=train_mode)
    if not device_type:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device=torch.device(device_type))
