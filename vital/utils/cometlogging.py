import configparser
import os
import warnings
from os.path import join as pjoin
from pathlib import Path
from os.path import isfile, dirname

from comet_ml import API


def load_or_download_weigths(name: Path, download_path: Path = Path('./')) -> Path:
    """Download weights from Comet-ML model registry if necessary.

    Args:
        name: name of the weights to download of load
        download_path: where to download the weights

    Returns:
        path to weights
    """
    if '.ckpt' in str(name):
        weights_path = name
    else:
        path = download_path / name
        if not os.path.isdir(path):
            comet_config = find_config_file(os.getcwd())  # Find .comet.config file if it exists
            config = read_comet_config(comet_config)
            api = API(api_key=config['api_key'])
            api.download_registry_model(config['workspace'], name, output_path=path, version='1.0.0')

        ckp_files = os.listdir(path)
        weights_path = os.path.join(path, ckp_files[0])

    return weights_path


def find_config_file(path, depth=5):
    """Recursively find comet config file."""
    if isfile(pjoin(path, '.comet.config')):
        return pjoin(path, '.comet.config')
    if depth == 0:
        warnings.warn("Did not find '.comet.config'")
        return None
    else:
        return find_config_file(dirname(path), depth=depth - 1)


def read_comet_config(comet_config: Path):
    """Read comet config file."""
    config = configparser.ConfigParser()
    config.read(str(comet_config))
    return dict(config["comet"])
