from pathlib import Path
from typing import Tuple

import SimpleITK
import numpy as np


def load_mhd(filepath: Path) -> Tuple[np.ndarray, Tuple[Tuple[int, ...]]]:
    """ This function loads a mhd image and the image and its metadata.

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
