from typing import Literal, TypeVar

import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity, NearestNeighbors
from torch import Tensor

from vital.utils.decorators import auto_cast_data

T = TypeVar("T", np.ndarray, Tensor)


@auto_cast_data
def estimate_kde_bandwidth(
    data: T,
    method: Literal[
        "silverman", "cross_validation", "scott_scipy", "silverman_scipy", "knn_mean_dist"
    ] = "knn_mean_dist",
    n_neighbors: int = 5,
) -> T:
    """Compute an estimate of the optimal KDE bandwidth for the provided data.

    Args:
        data: (N, D), Data points to estimate from.
        method: Method used to estimate the bandwidth. This can one of the following:
            - 'silverman': Anisotropic (i.e. feature-wise) implementation of Silverman's rule-of-thumb.
              (see ref: https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator)
            - 'cross_validation': Experimental cross-validation to search for the bandwidth that provides the best fit.
            - 'scott_scipy': Scott's rule, as formulated by `scipy` as part of the `gaussian_kde` API.
              (see ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
            - 'silverman_scipy': Silverman's rule, as formulated by `scipy` as part of the `gaussian_kde` API.
              (see ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
            - 'knn_mean_dist': Mean Euclidean distance between each point and their k-nearest neighbors.
        n_neighbors: If using the 'knn_mean_dist' method, the number of nearest neighbors to consider.

    Returns:
        (1,) or (N,), Isotropic or anisotropic bandwidth estimated from the data.
    """
    n, d = data.shape
    match method:
        case "silverman":
            sigma = np.std(data, axis=0)
            q1, q3 = np.quantile(data, [0.25, 0.75], axis=0)
            iqr = q3 - q1  # InterQuartile Range
            bw = 0.9 * np.minimum(sigma, iqr / 1.34) * (n ** (-0.2))
        case "cross_validation":
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"), {"bandwidth": 10 ** np.linspace(-1, 1, 100)}, cv=ShuffleSplit()
            )
            grid.fit(data)
            bw = grid.best_estimator_.bandwidth_
        case "scott_scipy":
            bw = n ** (-1 / (d + 4))
        case "silverman_scipy":
            bw = (n * (d + 2) / 4) ** (-1 / (d + 4))
        case "knn_mean_dist":
            neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
            # Get distance between each indexed points and its nearest neighbors (from the other indexed points)
            neigh_dist, neigh_ind = neigh.kneighbors()
            # Estimated the bandwidth as the mean distance between k nearest neighbors
            bw = neigh_dist.mean()
        case _:
            raise ValueError(
                f"Unexpected value for 'method': {method}. Use one of: ['silverman', 'cross_validation', "
                f"'scott_scipy', 'silverman_scipy', 'knn_mean_dist']."
            )
    return bw
