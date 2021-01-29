import torch
import torchvision.transforms.functional as F
from torch import Tensor


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
