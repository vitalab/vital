import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Callable, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback

from vital.results.processor import ResultsProcessor, ResultsProcessorCallback

logger = logging.getLogger(__name__)


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


def instantiate_config_node_leaves(
    cfg: DictConfig, node_desc: str, instantiate_fn: Callable[[DictConfig, str], Any] = None
) -> List[Any]:
    """Iterates over the leafs of the `cfg` config node and instantiates the leaves.

    Args:
        cfg: Root node whose leaves are to be instantiated.
        node_desc: Description of the node, used to display relevant messages in the logs. If not provided,
            defaults to `node_name`.
        instantiate_fn: Callback that instantiates an object from the config. If not provided, will default to call
            `hydra.utils.instantiate`.

    Returns:
        Objects instantiated from the leaves of the `cfg` config node.
    """
    if not instantiate_fn:
        instantiate_fn = hydra.utils.instantiate

    objects = []
    for obj_name, obj_cfg in cfg.items():
        if "_target_" in obj_cfg:
            logger.info(f"Instantiating {node_desc} <{obj_name}>")
            instantiate_args = []
            if instantiate_fn != hydra.utils.instantiate:  # If using a custom instantiation function
                instantiate_args = [obj_name]
            objects.append(instantiate_fn(obj_cfg, *instantiate_args))
        else:
            logger.warning(f"No '_target_' field in {node_desc} config. Cannot instantiate {obj_name}")
    return objects


def instantiate_results_processor(cfg: DictConfig, cfg_name: str) -> Callback:
    """Provides boilerplate for instantiating potential `ResultsProcessor`s as Lightning callbacks.

    Args:
        cfg: Config of the object to instantiate.
        cfg_name: Name of the config node to instantiate, used to display relevant messages in the logs.

    Returns:
        Either a Lightning callback directly, or `ResultsProcessor` wrapped as a Lightning callback.
    """
    processor = hydra.utils.instantiate(cfg)
    if isinstance(processor, Callback):
        # If the processor is already a Lightning `Callback`, leave it as is
        pass
    elif isinstance(processor, ResultsProcessor):
        # If the processor is a `ResultsProcessor`, use the generic callback wrapper
        processor = ResultsProcessorCallback(processor)
    else:
        raise ValueError(
            f"Unsupported type '{type(processor)}' for result processor <{cfg_name}>. It should be either a "
            f"'{ResultsProcessor}' or a '{Callback}'."
        )
    return processor
