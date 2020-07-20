from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Tuple, Type

import torch
from roi_align import CropAndResize
from torch import Tensor, nn
from torch.nn import functional as F

from vital.utils.image.transform import resize_image


class LocalizationNet(nn.Module):
    """Generalization of the LU-Net model, initially implemented for the CAMUS dataset.

    # TODO Add reference to paper describing the network

    This implementation generalizes the idea of the network beyond its U-Net roots. It makes it applicable to any
    segmentation model, given that it complies to the following interface:
        - Implemented as an nn.Module (therefore callable for making predictions)
        - Accepts the following kwargs in its constructor:
            - in_channels
            - out_channels
        - Provides the following attributes:
            - encoder
            - bottleneck
            - decoder
        - When ``encoder`` is called, returns either a single tensor to be passed to the bottleneck, or a tuple of
          tensors, where the expected input for the bottleneck is the first element of the tuple.
    """

    def __init__(
        self, segmentation_cls: Type[nn.Module], in_shape: Tuple[int, ...], out_shape: Tuple[int, ...], **kwargs
    ):
        """
        Args:
            segmentation_cls: class of the module to use as a base segmentation model for the LocalizationNet.
            data_shape: information about the shape of the expected input and output.
            **kwargs: arguments to initialize an instance of ``segmentation_cls``.
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.global_segmentation_module = segmentation_cls(
            in_channels=in_shape[-1], out_channels=out_shape[-1], **kwargs
        )
        self.localized_segmentation_module = segmentation_cls(
            in_channels=in_shape[-1], out_channels=out_shape[-1], **kwargs
        )
        segmentation_module = segmentation_cls(in_channels=out_shape[-1], out_channels=out_shape[-1], **kwargs)
        self.segmentation_encoder = segmentation_module.encoder
        self.segmentation_bottleneck = segmentation_module.bottleneck

        self.crop_resize = CropAndResize(*in_shape[:2])

        # Compute forward pass with dummy data to compute the output shape of the feature extractor module
        # used by the ROI bbox module (batch_size of 2 for batchnorm)
        input_size = (out_shape[-1], *out_shape[:2])
        x = torch.rand(2, *input_size).type(torch.float)
        features = self.segmentation_encoder(x)
        if isinstance(features, Tuple):  # In case of multiple tensors returned by the encoder
            features = features[0]  # Extract expected bottleneck input
        features = self.segmentation_bottleneck(features)
        self.roi_bbox_module = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(reduce(mul, features.shape[1:]), 1024)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(1024, 256)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("linear3", nn.Linear(256, 32)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("bbox_", nn.Linear(32, 4)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), input image to segment.

        Returns:
            global_y_hat: (N, ``out_channels``, H, W), raw, unnormalized scores for each class in the first, global
                          segmentation.
            roi_bbox_hat: (N, 4), coordinates of the bbox around the ROI.
            localized_y_hat: (N, ``out_channels``, H, W), raw, unnormalized scores for each class in the second
                             segmentation, localized in the cropped ROI.
        """
        # First segmentation module: Segment input image
        # Segmentation module trained to take as input the complete image, and to predict a rough segmentation from
        # which the groundtruth segmentation's ROI can be inferred
        global_y_hat = self.global_segmentation_module(x)

        # Feature extraction and bbox module: Regress the bbox coordinates around the ROI
        # Downsampling half of segmentation module trained, in association with the following fully-connected layers,
        # to predict through regression the coordinates of the bbox around the groundtruth segmentation
        features = self.segmentation_encoder(F.softmax(global_y_hat, dim=1))
        if isinstance(features, Tuple):  # In case of multiple tensors returned by the encoder
            features = features[0]  # Extract expected bottleneck input
        features = self.segmentation_bottleneck(features)
        features = torch.flatten(features, 1)  # Format bottleneck output to match expected bbox module input
        roi_bbox_hat = self.roi_bbox_module(features)

        # Crop and resize ``x`` based on ``roi_bbox_hat`` predicted by the previous modules
        # The ``torch.int32`` is important to pass internal assertions about the type of the bbox index
        cropped_x = self.crop_resize(x, roi_bbox_hat, torch.arange(x.shape[0], dtype=torch.int32, device=x.device))

        # Second segmentation module: Segment cropped ROI
        # Segmentation module trained to take as input the image cropped around the predicted segmentation's ROI, and
        # to predict a highly accurate segmentation from the localised input
        localized_y_hat = self.localized_segmentation_module(cropped_x)

        return global_y_hat, roi_bbox_hat, localized_y_hat

    def predict(self, x: Tensor) -> Tensor:
        """Performs test-time inference on the input.

        Args:
            x: (N, ``in_channels``, H, W), input image to segment.

        Returns:
            prediction: (N, ``out_channels``, H, W), input's segmentation, in one-hot format.
        """
        _, roi_bbox_hat, localized_y_hat = self(x)
        y_hat = self._revert_crop(localized_y_hat.argmax(dim=1, keepdim=True), roi_bbox_hat).squeeze()  # (N, H, W)
        y_hat = F.one_hot(y_hat.squeeze(), num_classes=self.out_shape[-1])  # (N, H, W, ``out_channels``)
        return y_hat.permute(0, 3, 1, 2)  # (N, ``out_channels``, H, W)

    def _revert_crop(self, localized_segmentation: Tensor, roi_bbox: Tensor) -> Tensor:
        """Fits the localized segmentation back to its original position the image.

        Args:
            localized_segmentation: (N, 1, H, W), segmentation of the content of the bbox around the ROI.
            roi_bbox: (N, 4), normalized coordinates of the bbox around the ROI.

        Returns:
            segmentation: (N, 1, H, W), localized segmentation fitted to its original position in the image.
        """
        # Clamp predicted normalized ROI bbox to ensure it won't end up out of range
        roi_bbox = roi_bbox.clamp(0, 1)

        # Change ROI bbox from normalized between 0 and 1 to absolute pixel coordinates
        roi_bbox[:, (0, 2)] = torch.round(roi_bbox[:, (0, 2)] * self.in_shape[0])  # Height
        roi_bbox[:, (1, 3)] = torch.round(roi_bbox[:, (1, 3)] * self.in_shape[1])  # Width
        roi_bbox = roi_bbox.int()

        # Fit the localized segmentation at its original location in the image, one item at a time
        segmentation = []
        for item_roi_bbox, item_localized_seg in zip(roi_bbox, localized_segmentation):
            # Get bbox size in order (width, height)
            bbox_size = (item_roi_bbox[3] - item_roi_bbox[1], item_roi_bbox[2] - item_roi_bbox[0])

            # Convert segmentation tensor to array (compatible with PIL) to resize, then convert back to tensor
            pil_formatted_localized_seg = item_localized_seg.byte().cpu().numpy().squeeze()
            item_resized_seg = torch.from_numpy(resize_image(pil_formatted_localized_seg, bbox_size)).unsqueeze(0)

            # Place the resized localised segmentation inside an empty segmentation
            segmentation.append(torch.zeros_like(item_localized_seg))
            segmentation[-1][
                :, item_roi_bbox[0] : item_roi_bbox[2], item_roi_bbox[1] : item_roi_bbox[3]
            ] = item_resized_seg

        return torch.stack(segmentation)
