import torch
import torchvision.transforms.functional as F


class NormalizeSample(torch.nn.Module):
    """Normalize a tensor image with its mean and standard deviation.

    Args:
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        """Apply normalization to tensor.

        Args:
            tensor (Tensor): Tensor image of size (1, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, [float(tensor.mean())], [float(tensor.std())], self.inplace)
