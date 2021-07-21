from pathlib import Path
from typing import Union

from comet_ml import API
from packaging.version import InvalidVersion, Version

from vital.utils.config import read_ini_config


def resolve_model_ckpt_path(ckpt: Union[str, Path], comet_config: Path = None, log_dir: Path = None) -> Path:
    """Resolves a local path or a Comet model registry query to a local path on the machine.

    Args:
        ckpt: Location of the checkpoint. This can be either a local path, or the fields of a query to a Comet model
            registry (e.g. using a version number: 'unet/0.1.0', or using a stage tag: 'unet/dev').
        comet_config: Path to Comet configuration file, used only if `ckpt` is a query to Comet's model registry.
        log_dir: Root path under which to download the model, used only if `ckpt` is a query to Comet's model registry.

    Returns:
        Path to the model's checkpoint file on the local computer. This can either be the `ckpt` already provided, if it
        was already local, or the location where the checkpoint was downloaded, if it pointed to a Comet registry model.
    """
    if ckpt.suffix == ".ckpt":
        local_ckpt_path = ckpt
    else:
        if comet_config is None:
            raise ValueError(
                f"The format of the checkpoint '{ckpt}' indicates you want to download the checkpoint off a Comet "
                f"model registry, but you have provided no Comet configuration to use. Either switch to providing a "
                f"local checkpoint path, or indicate a Comet configuration to use."
            )
        if log_dir is None:
            raise ValueError(
                f"The format of the checkpoint '{ckpt}' indicates you want to download the checkpoint off a Comet "
                f"model registry, but you have provided no root path where to save the downloaded model. Either switch "
                f"to providing a local checkpoint path, or indicate a directory where to download the model."
            )
        config = read_ini_config(comet_config)["comet"]
        api = API(api_key=config["api_key"])

        # Parse the provided checkpoint path as a query for a Comet model registry
        version, stage = None, None
        if len(ckpt.parts) == 2:
            registry_name, version_or_stage = ckpt.parts
            try:
                Version(version_or_stage)  # Will fail if `version_or_stage` cannot be parsed as a version
                version = version_or_stage
            except InvalidVersion:
                stage = version_or_stage
        else:
            raise ValueError(f"Failed to interpret checkpoint '{ckpt}' as a query for a Comet model registry.")

        # Download the Comet registry model local and extracts its local path
        download_path = log_dir / ckpt
        api.download_registry_model(
            config["workspace"], registry_name, version=version, stage=stage, output_path=str(download_path)
        )
        local_ckpt_path = list(download_path.iterdir())[0]

    return local_ckpt_path
