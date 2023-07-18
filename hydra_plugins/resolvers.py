import builtins
import operator
import os
from typing import Any

import numpy as np
from omegaconf import ListConfig, OmegaConf


# Define wrapper for basic math operators, with the option to cast result to arbitrary type
def _cast_op(op, x, y, type_of: str = None) -> Any:
    res = op(x, y)
    if type_of is not None:
        res = getattr(builtins, type_of)(res)
    return res


# System information resolvers
OmegaConf.register_new_resolver("sys.num_workers", lambda x=None: os.cpu_count() - 1)
OmegaConf.register_new_resolver("sys.getcwd", lambda x=None: os.getcwd())
OmegaConf.register_new_resolver("sys.eps.np", lambda dtype: np.finfo(np.dtype(dtype)).eps)

# Builtin operators resolvers
OmegaConf.register_new_resolver("op.add", lambda x, y, type_of=None: _cast_op(operator.add, x, y, type_of=type_of))
OmegaConf.register_new_resolver("op.sub", lambda x, y, type_of=None: _cast_op(operator.sub, x, y, type_of=type_of))
OmegaConf.register_new_resolver("op.mul", lambda x, y, type_of=None: _cast_op(operator.mul, x, y, type_of=type_of))

# Builtin functions resolvers
OmegaConf.register_new_resolver("builtin.len", lambda cfg: len(cfg))
OmegaConf.register_new_resolver("builtin.range", lambda start, stop, step=1: list(range(start, stop, step)))

# Data structs utils resolvers
OmegaConf.register_new_resolver(
    "list.remove",
    lambda cfg, to_remove: ListConfig(
        [
            val
            for val in cfg
            if (val not in to_remove if isinstance(to_remove, (tuple, list, ListConfig)) else val != to_remove)
        ]
    ),
)
