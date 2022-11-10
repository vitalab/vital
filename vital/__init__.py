import os
from pathlib import Path

from dotenv import load_dotenv

ENV_VITAL_HOME = "VITAL_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def get_vital_root() -> Path:
    """Resolves the root directory for the `vital` package.

    Returns:
        Path to the root directory for the `vital` package.
    """
    return Path(__file__).resolve().parent


def get_vital_home() -> Path:
    """Resolves the home directory for the `vital` library, used to save/cache data reusable across scripts/runs.

    Returns:
        Path to the home directory for the `vital` library.
    """
    load_dotenv()
    vital_home = os.getenv(ENV_VITAL_HOME)
    if vital_home is None:
        user_cache_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_CACHE_DIR)
        vital_home = os.path.join(user_cache_dir, "vital")
    return Path(vital_home).expanduser()
