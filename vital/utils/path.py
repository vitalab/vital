from pathlib import Path


def remove_suffixes(filename: Path) -> Path:
    """Removes all the suffixes from `filename`, unlike `stem` which only removes the last suffix.

    Args:
        filename: Filename from which to remove all extensions.

    Returns:
        Filename without its extensions.
    """
    return Path(str(filename).removesuffix("".join(filename.suffixes)))
