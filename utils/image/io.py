from pathlib import Path
from typing import Tuple

import SimpleITK
import numpy as np


def load_and_process_mhd_data(filename: str) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """ This function loads a mhd image and the image and its metadata.

    Args:
        filename: path to the image.

    Returns:
        - ([N], H, W, C), image array
        - collection of metadata.
    """

    # load image and save info
    image = SimpleITK.ReadImage(filename)
    info = (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # create numpy array from the .mhd file and corresponding image
    im_array = np.squeeze(SimpleITK.GetArrayFromImage(image))

    return im_array, info


def save_as_mhd(im_array: np.ndarray, output_filepath: Path, origin=(0, 0, 0), spacing=(1, 1, 1), dtype=np.float32):
    """ Saves an array to mhd format.

    Args:
        im_array: the 2d or 3d image.
        output_filepath: the output filename. Must end in ".mhd".
        origin: center of the image.
        spacing: size of the voxels along each dimension.
        dtype: type of data to save.
    """
    seg = SimpleITK.GetImageFromArray(im_array.astype(dtype))
    seg.SetOrigin(origin)
    seg.SetSpacing(spacing)
    SimpleITK.WriteImage(seg, str(output_filepath))
