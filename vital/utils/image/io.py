from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageSequence


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
