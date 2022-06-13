from configparser import ConfigParser
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def read_ini_config(ini_config: Path) -> ConfigParser:
    """Reads all values from an ini configuration file.

    Args:
        ini_config: Path of the ini configuration file to read.

    Returns:
        Two-tiered mapping, with section names as first level keys and value keys as second level keys.
    """
    config = ConfigParser()
    config.read(str(ini_config))
    return config


def ascend_config_node(cfg: DictConfig, node: str) -> DictConfig:
    """Ascends unwanted node introduced in config hierarchy by directories meant to keep config files organised.

    If `cfg` is None or `node` does not exist, simply return `cfg` without modifying it.

    Args:
        cfg: Config node with an unwanted child node to bring to the top-level.
        node: Name of the node to bring back to the top level of `cfg`.

    Returns:
        `cfg` with the content of `node` merged at its top-level.
    """
    if isinstance(cfg, DictConfig) and node in cfg:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict.update(cfg_dict.pop(node))
        cfg = OmegaConf.create(cfg_dict)
    return cfg
