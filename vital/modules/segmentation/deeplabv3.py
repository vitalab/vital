from typing import Literal

import torchvision
from torch import Tensor, nn

from vital.data.transforms import GrayscaleToRGB


class DeepLabv3(nn.Module):
    """Wrapper around torchvision's implementation of the DeepLabv3 model that allows for single-channel inputs."""

    def __init__(
        self, backbone: Literal["resnet50", "resnet101"], num_classes: int, convert_grayscale_to_rgb: bool = False
    ):  # noqa: D205,D212,D415
        """
        Args:
            backbone: The network used by the DeepLabv3 architecture to compute the features for the model.
            num_classes: Number of output classes to segment.
            convert_grayscale_to_rgb: If ``True``, the forward pass will automatically convert single channel grayscale
                inputs to 3-channel RGB, where r==g==b, to fit with DeepLabv3's hardcoded 3 channel input layer.
                If ``False``, the input is assumed to already be 3 channel and is not transformed in any way.
        """
        super().__init__()
        self._convert_grayscale_to_rgb = convert_grayscale_to_rgb
        if self._convert_grayscale_to_rgb:
            self._grayscale_trans = GrayscaleToRGB()
        module_cls = torchvision.models.segmentation.__dict__[f"deeplabv3_{backbone}"]
        self._network = module_cls(pretrained=False, aux_loss=False, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        if self._convert_grayscale_to_rgb and x.shape[1] != 3:
            x = self._grayscale_trans(x)
        return self._network(x)["out"]
