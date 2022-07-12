from typing import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter

from vital.utils.format.numpy import to_onehot
from vital.utils.image.process import PostProcessor


class PostGaussianFilter(PostProcessor):
    """Post-processing that applies a gaussian filter along specific dimensions of the segmentation maps."""

    def __init__(
        self,
        sigmas: Mapping[int, float],
        sigmas_as_ratio: bool = False,
        extend_modes: Mapping[int, str] = None,
        do_argmax: bool = True,
    ):
        """Initializes class instance.

        Args:
            sigmas: Mapping between dimension(s) along which to apply the filter, and the standard deviation to use for
                that dimension's Gaussian kernel.
            sigmas_as_ratio: Whether to interpret `sigmas` relative to the length of the input along that dimension, so
                that the true standard deviation for a given `dim` becomes `sigma * input.shape[dim]`.
            extend_modes: Mapping between dimension(s) along which to apply the filter, and the mode to use to extend
                the input when the filter overlaps a border.
            do_argmax: If `True`, pass the filter's result through an argmax to restore "hard" class labels. Otherwise,
                returns directly the "soft" class probabilities computed by the Gaussian filter, in a new dimension
                added last.
        """
        self._kernel_stddevs = sigmas
        self._scale_stddevs = sigmas_as_ratio
        self._extend_modes = extend_modes if extend_modes else {}
        self._do_argmax = do_argmax

    def __call__(self, seg: np.ndarray, **kwargs) -> np.ndarray:
        """Applies a gaussian filter along specific dimensions of the segmentation maps.

        Args:
            seg: (N, H, W), Segmentation to process.
            **kwargs: Capture non-used parameters to get a callable API compatible with similar callables.

        Returns:
            (N, H, W, [C]), Processed segmentation, with an additional last dimension for the probability of each class
            if `do_argmax=False`.
        """
        onehot_seg = to_onehot(seg)
        sigmas = np.array([self._kernel_stddevs.get(dim, 0) for dim in range(onehot_seg.ndim)])
        if self._scale_stddevs:
            sigmas = sigmas * onehot_seg.shape
        modes = [self._extend_modes.get(dim, "reflect") for dim in range(onehot_seg.ndim)]
        filtered_seg = gaussian_filter(onehot_seg, sigmas, output=float, mode=modes)
        if self._do_argmax:
            filtered_seg = filtered_seg.argmax(axis=-1).astype(seg.dtype)
        return filtered_seg
