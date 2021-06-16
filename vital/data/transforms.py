import numpy as np
import torch
import torchvision.transforms.functional as F
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


class RescaleIntensity(torch.nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def __call__(self, tensor):
        if tensor.max() < 1e-6:
            return tensor
        return (
            (tensor - tensor.min())
            * (self.max - self.min)
            / (tensor.max() - tensor.min())
        ) + self.min