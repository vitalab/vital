import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor

from vital.utils.image.transform import segmentation_to_tensor


class NormalizeSample(torch.nn.Module):
    """Normalize a tensor image with its mean and standard deviation.

    Args:
        inplace: Whether to make this operation in-place.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor) -> Tensor:
        """Apply normalization to tensor.

        Args:
            tensor: image of size (1, H, W) to be normalized.

        Returns:
            Normalized image.
        """
        return F.normalize(tensor, [float(tensor.mean())], [float(tensor.std())], self.inplace)


class PreQuantilePercent(torch.nn.Module):
    """Quantile normalization class.

    Args:
            percent: int, Above this percent the data are considered outliers.
    """

    def __init__(self, percent: float = 0.96):
        super().__init__()
        self.percent = percent

    def __call__(self, tensor: Tensor) -> Tensor:
        """Processing the percentile normalization.

        Set all outlier values higher than a given percentage to the highest
        acceptable value.

        Args:
            tensor: data to apply the preprocessing.

        Returns:
            Data without outlier
        """
        tresh = torch.quantile(tensor, self.percent)
        idx = tensor > tresh
        tensor[idx] = tensor.min()
        tensor[idx] = tensor.max()
        return tensor


class SegmentationToTensor(torch.nn.Module):
    """Converts a segmentation map to a tensor."""

    def __call__(self, data: np.ndarray) -> Tensor:
        """Apply conversion.

        Args:
            segmentation: ([N], H, W, [C]), Segmentation map to convert to a tensor.

        Returns:
            Converted segmentation map
        """
        return segmentation_to_tensor(data)
