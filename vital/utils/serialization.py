import logging
import shutil
from pathlib import Path
from typing import Union

from comet_ml import API
from packaging.version import InvalidVersion, Version

from vital import get_vital_home
from vital.utils.config import read_ini_config

logger = logging.getLogger(__name__)


def resolve_model_ckpt_path(ckpt: Union[str, Path], comet_config: Path = None) -> Path:
    """Resolves a local path or a Comet model registry query to a local path on the machine.

    Args:
        ckpt: Location of the checkpoint. This can be either a local path, or the fields of a query to a Comet model
            registry (e.g. using a version number: 'unet/0.1.0', or using a stage tag: 'unet/dev').
        comet_config: Path to Comet configuration file, used only if `ckpt` is a query to Comet's model registry.

    Returns:
        Path to the model's checkpoint file on the local computer. This can either be the `ckpt` already provided, if it
        was already local, or the location where the checkpoint was downloaded, if it pointed to a Comet registry model.
    """
    ckpt = Path(ckpt)
    if ckpt.suffix == ".ckpt":
        local_ckpt_path = ckpt
    else:
        if comet_config is None:
            raise ValueError(
                f"The format of the checkpoint '{ckpt}' indicates you want to download the checkpoint off a Comet "
                f"model registry, but you have provided no Comet configuration to use. Either switch to providing a "
                f"local checkpoint path, or indicate a Comet configuration to use."
            )
        config = read_ini_config(comet_config)["comet"]
        api = API(api_key=config["api_key"])

        # Parse the provided checkpoint path as a query for a Comet model registry
        version_or_stage, version, stage = None, None, None
        if len(ckpt.parts) == 1:
            (registry_name,) = ckpt.parts
        elif len(ckpt.parts) == 2:
            registry_name, version_or_stage = ckpt.parts
            try:
                Version(version_or_stage)  # Will fail if `version_or_stage` cannot be parsed as a version
                version = version_or_stage
            except InvalidVersion:
                stage = version_or_stage
        else:
            raise ValueError(f"Failed to interpret checkpoint '{ckpt}' as a query for a Comet model registry.")

        # If neither version nor stage were provided, use latest version available
        if not version_or_stage:
            version_or_stage = version = api.get_registry_model_versions(config["workspace"], registry_name)[-1]

        # Determine where to download the checkpoint locally
        cache_dir = get_vital_home()
        model_cached_path = cache_dir / config["workspace"] / registry_name / version_or_stage

        # When using stage, delete cached versions and force re-downloading the registry model,
        # because stage tags can be changed
        if stage:
            shutil.rmtree(model_cached_path, ignore_errors=True)

        # Download model if not already cached
        if not model_cached_path.exists():
            api.download_registry_model(
                config["workspace"], registry_name, version=version, stage=stage, output_path=str(model_cached_path)
            )
        else:
            logger.info(
                f"Using cached registry model {registry_name}, version {version} from workspace {config['workspace']} "
                f"located in '{model_cached_path}'."
            )

        # Extract the path of the checkpoint file on the local machine
        local_ckpt_path = list(model_cached_path.iterdir())[0]

    return local_ckpt_path
