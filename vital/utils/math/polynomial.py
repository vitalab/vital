import numpy as np
from numpy.polynomial import Polynomial


def polyfit(x: np.ndarray, y: np.ndarray, degree: int = 1, padding: float = None) -> np.ndarray:
    """Fits a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` to points (x,y).

    Args:
        x: Independent variable for the polynomial function(s).
        y: Dependent variable(s) for the polynomial function(s).
        degree: Degree of the fitting polynomial.
        padding: In case of cyclical data to fit, the fraction of the data, i.e. [0,1], to repeat before/after the data
            (before fitting the polynomial) to smooth the transitions at the beginning/end of the cycle.

    Returns:
        The dependent variable(s) of the fitted polynomial function(s).
    """
    if padding:
        num_items_to_repeat = int(len(x) * padding)
        slice_begin = np.s_[1 : num_items_to_repeat + 1]
        slice_end = np.s_[-num_items_to_repeat - 1 : -1]
        # Pad independent variable before/after with values with equal diffs to first/last than end/begin have to
        # last/first
        x = np.hstack((x[0] + x[slice_end] - x[-1], x, x[-1] + x[slice_begin] - x[0]))
        # Pad dependent variable before/after with values repeated from end/begin
        y = np.hstack((y[slice_end], y, y[slice_begin]))
    fit = Polynomial.fit(x, y, degree)(x)
    if padding:
        fit = fit[num_items_to_repeat:-num_items_to_repeat]
    return fit
