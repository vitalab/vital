import functools

import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy import interpolate, signal
from torch import Tensor

from vital.utils.image.transform import segmentation_to_tensor


class NormalizeSample(torch.nn.Module):
    """Normalizes a tensor w.r.t. to its mean and standard deviation.

    Args:
        inplace: Whether to make this operation in-place.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor) -> Tensor:
        """Normalizes input tensor.

        Args:
            tensor: Tensor to normalize.

        Returns:
            Normalized image.
        """
        return F.normalize(tensor, [float(tensor.mean())], [float(tensor.std())], self.inplace)


class SegmentationToTensor(torch.nn.Module):
    """Converts a segmentation map to a tensor."""

    def __call__(self, data: np.ndarray) -> Tensor:
        """Converts the segmentation map to a tensor.

        Args:
            segmentation: ([N], H, W), Segmentation map to convert to a tensor.

        Returns:
            ([N], H, W), Segmentation map converted to a tensor.
        """
        return segmentation_to_tensor(data)


class GrayscaleToRGB(torch.nn.Module):
    """Converts grayscale image to RGB image where r == g == b."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Converts grayscale image to RGB image where r == g == b.

        Args:
            img: (N, 1, ...), Grayscale image to convert to RGB.

        Returns:
            (N, 3, ...), RGB version of the original grayscale image, where r == g == b.
        """
        if img.shape[1] == 1:
            repeat_sizes = [1] * img.ndim
            repeat_sizes[1] = 3
            img = img.repeat(*repeat_sizes)
        else:
            raise ValueError(
                f"{self.__class__.__name__} only supports converting single channel grayscale images to RGB images "
                f"where r == g == b. The image data you provided consists of {img.shape[1]} channel images."
            )
        return img


class Resample:
    """Resamples a signal to reach a target number of data points."""

    def __init__(self, num: int, **resample_kwargs):
        """Initializes class instance.

        Args:
            num: Required parameter to pass along to ``scipy.signal.resample``, indicating the number of samples in the
                resampled signal.
            **resample_kwargs: Additional parameters to pass along to ``scipy.signal.resample``.
        """
        super().__init__()
        self.partial_resample = functools.partial(signal.resample, num=num, **resample_kwargs)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Resamples input signal.

        Args:
            signal: (M), Signal to resample.

        Returns:
            (N), Resampled signal.
        """
        return self.partial_resample(signal)


class Interp1d:
    """Interpolates data points in a signal to reach a target number of data points."""

    def __init__(self, num: int, **interp1d_kwargs):
        """Initializes class instance.

        Args:
            num: Number of samples to interpolate from the original signal.
            **interp1d_kwargs: Additional parameters to pass along to ``scipy.signal.resample``.
        """
        super().__init__()
        self.interp_x_coords = np.linspace(0, 1, num=num)
        self.interp1d_kwargs = interp1d_kwargs

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Interpolates input signal.

        Args:
            signal: (M), Signal to interpolate.

        Returns:
            (N), Interpolated signal.
        """
        signal_x_coords = np.linspace(0, 1, num=len(signal))
        f = interpolate.interp1d(signal_x_coords, signal, **self.interp1d_kwargs)
        return f(self.interp_x_coords)


class PadShift1d:
    """Shift the signal forward/backward after padding in the direction of the shift."""

    def __init__(self, shift: int, before_shift_prob: float = 0.5, **pad_kwargs):
        """Initializes class instance.

        Args:
            shift: Number of values by which to shift the signal.
            before_shift_prob: Probability to shift the signal backward (to include values padded before the signal).
            **pad_kwargs: Additional parameters to pass along to ``np.pad``.
        """
        super().__init__()
        self.shift = shift
        self.pad_kwargs = pad_kwargs
        self.before_shift_prob = before_shift_prob

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Shifts input signal.

        Args:
            signal: (N), Signal to shift.

        Returns:
            (N), Shifted signal.
        """
        if signal.ndim != 1:
            raise ValueError(
                f"The provided data on which to apply the '{self.__class__.__name__}' transform does not match the "
                f"configured shift. The '{self.shift}' shift is configured for 1D signal, but the provided data is "
                f"{signal.ndim}D with a shape of {signal.shape}."
            )

        padded_signal = np.pad(signal, (self.shift, self.shift), **self.pad_kwargs)

        shift_before = np.random.binomial(1, self.before_shift_prob)
        if shift_before:  # Start from the values padded before `data` and exclude the last values of `data`
            shifted_signal = padded_signal[: len(signal)]
        else:  # Shift to include values padded after `data`
            shifted_signal = padded_signal[-len(signal) :]

        return shifted_signal
