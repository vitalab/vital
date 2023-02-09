from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from vital import get_vital_root
from vital.data.cardinal.utils.process import postprocess_views
from vital.utils.config import instantiate_config_node_leaves, register_omegaconf_resolvers


@hydra.main(version_base=None, config_path="../../config", config_name="experiment/cardinal-process_masks")
def process_masks(cfg: DictConfig):
    """Handles the processing to ensure the quality of segmentation masks predicted by a model.

    Args:
        cfg: Configuration to run the script.
    """
    hydra_cfg = HydraConfig.get()

    # Instantiate objects from Hydra config
    views = hydra.utils.instantiate(cfg.views)

    postprocess_ops = []
    if cfg.morphological_preprocessing:
        # Prepare a first pass o basic morphological post-processing operations before the configured post-processing
        morphological_cfg_file = get_vital_root() / "config/data/postprocessing/echo/default.yaml"
        morphological_cfg = OmegaConf.load(morphological_cfg_file)
        postprocess_ops.extend(instantiate_config_node_leaves(morphological_cfg, "morphological pre-processing"))
    # Add the configured post-processing
    postprocess_ops.extend(instantiate_config_node_leaves(cfg.postprocess_ops, "post-processing"))

    # Post-process views' segmentation masks and save the output to disk
    postprocess_views(
        views, cfg.mask_tag, postprocess_ops, Path(hydra_cfg.runtime.output_dir), progress_bar=cfg.progress_bar
    )


def main():
    """Run the script."""
    # Load environment variables from `.env` file if it exists
    # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
    load_dotenv()
    register_omegaconf_resolvers()

    process_masks()


if __name__ == "__main__":
    main()
