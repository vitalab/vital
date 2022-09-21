import importlib
from typing import Any


def import_from_module(dotpath: str) -> Any:
    """Dynamically imports an object from a module based on its "dotpath".

    Args:
        dotpath: "Dotpath" (name that can be looked up via importlib) where the firsts components specify the module to
            look up, and the very last component is the attribute to import from this module.

    Returns:
        Target object.
    """
    module, module_attr = dotpath.rsplit(".", 1)
    return getattr(importlib.import_module(module), module_attr)
