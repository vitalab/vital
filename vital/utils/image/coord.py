from typing import Tuple

import numpy as np


def cart2pol(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts (x,y) cartesian coordinates to (theta,rho) polar coordinates.

    Notes:
        - Made to mimic Matlab's `cart2pol` function: https://www.mathworks.com/help/matlab/ref/pol2cart.html

    Args:
        x: x component of cartesian coordinates.
        y: y component of cartesian coordinates.

    Returns:
        (theta,rho) polar coordinates corresponding to the input cartesian coordinates.

    Example:
        >>> x = np.array([5, 3.5355, 0, -10])
        >>> y = np.array([0, 3.5355, 10, 0])
        >>> cart2pol(x,y)
        (array([0, 0.7854, 1.5708, 3.1416]), array([5.0000, 5.0000, 10.0000, 10.0000]))
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho


def pol2cart(theta: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts (theta,rho) polar coordinates to (x,y) cartesian coordinates.

    Notes:
        - Made to mimic Matlab's `pol2cart` function: https://www.mathworks.com/help/matlab/ref/pol2cart.html

    Args:
        theta: Angular component of polar coordinates. `theta` is the counterclockwise angle in the x-y plane measured
            in radians from the positive x-axis.
        rho: Radial distance (distance to origin) of polar coordinates.

    Returns:
        (x,y) cartesian coordinates corresponding to the input polar coordinates.

    Example:
        >>> theta = np.array([0, np.pi/4, np.pi/2, np.pi])
        >>> rho = np.array([5, 5, 10, 10])
        >>> pol2cart(theta, rho)
        (array([5.0000, 3.5355, 0.0000, -10.0000]), array([0, 3.5355, 10.0000, 0.0000]))
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y
