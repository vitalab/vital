from pathlib import Path


def as_file_extension(extension: str) -> str:
    """Ensures the input string is a valid suffix, by adding a leading dot if necessary.

    Args:
        extension: String to ensure is a file extension.

    Returns:
        String, converted to a valid suffix by adding a leading dot if necessary.
    """
    if not extension.startswith("."):
        extension = "." + extension
    return extension


def remove_suffixes(filename: Path) -> Path:
    """Removes all the suffixes from `filename`, unlike `stem` which only removes the last suffix.

    Args:
        filename: Filename from which to remove all extensions.

    Returns:
        Filename without its extensions.
    """
    return Path(str(filename).removesuffix("".join(filename.suffixes)))
