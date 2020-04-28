from numbers import Number
from pathlib import Path
from typing import Tuple

import SimpleITK
import numpy as np


def load_mhd(filepath: Path) -> Tuple[np.ndarray, Tuple[Tuple[Number, ...], ...]]:
    """This function loads a mhd image and returns the image and its metadata.

    Args:
        filepath: path to the image.

    Returns:
        - ([N], H, W), image array.
        - collection of metadata.
    """

    # load image and save info
    image = SimpleITK.ReadImage(str(filepath))
    info = (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # create numpy array from the .mhd file and corresponding image
    im_array = np.squeeze(SimpleITK.GetArrayFromImage(image))

    return im_array, info


def save_as_mhd(im_array: np.ndarray, output_filepath: Path, origin=(0, 0, 0), spacing=(1, 1, 1), dtype=np.float32):
    """ Saves an array to mhd format.

    Args:
        im_array: ([N], H, W), image array.
        output_filepath: the output filename. Must end in ".mhd".
        origin: center of the image.
        spacing: size of the voxels along each dimension.
        dtype: type of data to save.
    """
    seg = SimpleITK.GetImageFromArray(im_array.astype(dtype))
    seg.SetOrigin(origin)
    seg.SetSpacing(spacing)
    SimpleITK.WriteImage(seg, str(output_filepath))
