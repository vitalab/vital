from numbers import Number
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK
from PIL import Image, ImageSequence


def sitk_load(filepath: Path) -> Tuple[np.ndarray, Tuple[Tuple[Number, ...], ...]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # load image and save info
    image = SimpleITK.ReadImage(str(filepath))
    info = (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # create numpy array from the .mhd file and corresponding image
    im_array = np.squeeze(SimpleITK.GetArrayFromImage(image))

    return im_array, info


def sitk_save(
    im_array: np.ndarray, output_filepath: Path, origin=(0, 0, 0), spacing=(1, 1, 1), dtype=np.float32
) -> None:
    """Saves an array to an image format using SimpleITK.

    Args:
        im_array: ([N], H, W), Image array.
        output_filepath: Output filename. Must end in ".mhd".
        origin: Center of the image.
        spacing: (W, H, N), Size of the voxels along each dimension. Be careful about the order of the dimensions,
            because it is not the same as the image array itself.
        dtype: Type of data to save.
    """
    seg = SimpleITK.GetImageFromArray(im_array.astype(dtype))
    seg.SetOrigin(origin)
    seg.SetSpacing(spacing)
    SimpleITK.WriteImage(seg, str(output_filepath))


def load_gif(filepath: Path) -> np.ndarray:
    """Loads an animated GIF image as a sequence of 2D images.

    Args:
        filepath: Path to the image.

    Returns:
        (T, H, W) array of the image's pixel values.
    """
    return np.array([np.array(frame) for frame in ImageSequence.Iterator(Image.open(filepath))])
