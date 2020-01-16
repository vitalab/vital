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
