import os

import numpy as np
import torch
from omegaconf import OmegaConf


def register_omegaconf_resolvers() -> None:
    """Registers various OmegaConf resolvers useful to query system info."""
    OmegaConf.register_new_resolver("sys.gpus", lambda x=None: int(torch.cuda.is_available()))
    OmegaConf.register_new_resolver("sys.num_workers", lambda x=None: os.cpu_count() - 1)
    OmegaConf.register_new_resolver("sys.getcwd", lambda x=None: os.getcwd())
    OmegaConf.register_new_resolver("sys.eps.np", lambda dtype: np.finfo(np.dtype(dtype)).eps)
