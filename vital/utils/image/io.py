from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageSequence


def check_image_io_format_support(format: str, io_mode: Literal["read", "write"], raise_err: bool = False) -> bool:
    """Checks if a specific `format` image extension is supported by SimpleITK image for either read/write operations.

    Args:
        format: Image format.
        io_mode: Target I/O mode for which to test if the format is supported by SimpleITK.
        raise_err: Whether to raise an error if the image format is not supported by SimpleITK.

    Returns:
        `True` if the image format is supported by SimpleITK for the target I/O mode, `False` otherwise.
    """
    match io_mode:
        case "read":
            image_io_formats = sitk.ImageFileReader().GetRegisteredImageIOs()
        case "write":
            image_io_formats = sitk.ImageFileWriter().GetRegisteredImageIOs()
        case _:
            raise ValueError(f"Unexpected value for 'io_mode': {io_mode}. Use one of: ['read', 'write'].")

    is_format_supported = format not in image_io_formats
    if raise_err and not is_format_supported:
        raise RuntimeError(
            f"Image format requested not supported by SimpleITK backend. Use one of the following supported "
            f"'{io_mode}' format instead: {image_io_formats}."
        )
    return is_format_supported


def sitk_load(filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))

    return im_array, info


def sitk_save(
    im_array: np.ndarray, output_filepath: Union[str, Path], origin=(0, 0, 0), spacing=(1, 1, 1), dtype=np.float32
) -> None:
    """Saves an array to an image format using SimpleITK.

    Args:
        im_array: ([N], H, W), Image array.
        output_filepath: Output filename, including the desired file extension.
        origin: Center of the image.
        spacing: (W, H, N), Size of the voxels along each dimension. Be careful about the order of the dimensions,
            because it is not the same as the image array itself.
        dtype: Type of data to save.
    """
    seg = sitk.GetImageFromArray(im_array.astype(dtype))
    seg.SetOrigin(origin)
    seg.SetSpacing(spacing)
    sitk.WriteImage(seg, str(output_filepath))


def load_gif(filepath: Path) -> np.ndarray:
    """Loads an animated GIF image as a sequence of 2D images.

    Args:
        filepath: Path to the image.

    Returns:
        (T, H, W) array of the image's pixel values.
    """
    return np.array([np.array(frame) for frame in ImageSequence.Iterator(Image.open(filepath))])
