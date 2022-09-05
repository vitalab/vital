import os

import numpy as np
from omegaconf import OmegaConf

from vital import get_vital_root


def register_omegaconf_resolvers() -> None:
    """Registers various OmegaConf resolvers useful to query system info."""
    OmegaConf.register_new_resolver("sys.num_workers", lambda x=None: os.cpu_count() - 1)
    OmegaConf.register_new_resolver("sys.getcwd", lambda x=None: os.getcwd())
    OmegaConf.register_new_resolver("sys.eps.np", lambda dtype: np.finfo(np.dtype(dtype)).eps)
    OmegaConf.register_new_resolver("vital.root", lambda x=None: str(get_vital_root()))
