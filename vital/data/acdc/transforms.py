import torchvision.transforms.functional as F


class NormalizeSample(object):

    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (1, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        return F.normalize(tensor, [float(tensor.mean())], [float(tensor.std())], self.inplace)

    def __repr__(self):
        return self.__class__.__name__
