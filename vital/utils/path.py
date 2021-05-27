import os
from pathlib import Path


def load_env_var(path: str):
    is_Path = False
    if isinstance(path, Path):
        is_Path = True
        path = str(path)

    if path.startswith('$'):
        path = os.environ.get(path[1:])

    path = Path(path) if is_Path else path
    return path
