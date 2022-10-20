from typing import Tuple, TypeVar, Union

import numpy as np
from torch import Tensor

T = TypeVar("T", np.ndarray, Tensor)


def minmax_scaling(data: T, bounds: Tuple[Union[float, T], Union[float, T]] = None) -> T:
    """Standardizes data w.r.t. predefined min/max bounds, if provided, or its own min/max otherwise.

    Args:
        data: Data to scale.
        bounds: Prior min and max bounds to use to scale the data.

    Returns:
        Data scaled w.r.t. the predefined or computed bounds.
    """
    # If no prior min/max bounds for the data are provided, compute them from its values
    if bounds is None:
        min, max = data.min(), data.max()
    else:
        min, max = bounds
    return (data - min) / (max - min)


def scale(data: T, bounds: Tuple[Union[float, T], Union[float, T]]) -> T:
    """Scales data that was previously normalized back to its original range, defined by min and max values in `bounds`.

    Args:
        data: Data to scale.
        bounds: Prior min and max bounds used to normalize the data originally.

    Returns:
        Data scaled w.r.t. the predefined bounds.
    """
    min, max = bounds
    return (data * (max - min)) + min
