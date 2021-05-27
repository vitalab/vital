import os

import hydra
from config.acdc import ACDCConfig
from config.data.acdc import AcdcConfig
from config.data.camus import CamusConfig
from config.data.mnist import MnistConfig
from config.default import DefaultConfig
from config.system.modules.mlp import MLPConfig
from config.system.modules.unet import UNetConfig
from config.system.segmentation import SegmentationConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"


def create_configs(cs: ConfigStore):
    cs.store(name="default", node=DefaultConfig)
    cs.store(name="acdc", node=ACDCConfig)


def store_groups(cs: ConfigStore):
    configuration = {
        "data": {"acdc": AcdcConfig, "camus": CamusConfig, "mnist": MnistConfig},
        "network": {"unet": UNetConfig, "mlp": MLPConfig},
        "system": {"segmentation": SegmentationConfig},
    }

    for group_name, group in configuration.items():
        for name, node in group.items():
            cs.store(group=group_name, name=name, node=node)


@hydra.main(config_name="default")
def my_app(cfg: DictConfig) -> None:
    print("Hello world")
    print(OmegaConf.to_yaml(cfg))

    datamodule = hydra.utils.instantiate(cfg.data)
    network = hydra.utils.instantiate(
        cfg.network, input_shape=datamodule.data_params.in_shape, ouput_shape=datamodule.data_params.out_shape
    )

    loss = hydra.utils.instantiate(cfg.system.loss)
    system = hydra.utils.instantiate(cfg.system, network, data_params=datamodule.data_params)

    print(datamodule)
    print(network)
    print(system)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    create_configs(cs)
    store_groups(cs)
    my_app()
