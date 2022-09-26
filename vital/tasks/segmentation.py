from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import roi_pool

from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.tasks.generic import SharedStepsTask
from vital.utils.decorators import auto_move_data
from vital.utils.image.measure import Measure


class SegmentationTask(SharedStepsTask):
    """Generic segmentation training and inference steps.

    Implements generic segmentation train/val step and inference, assuming the following conditions:
        - the model from ``self.configure_model()`` returns one output: the raw, unnormalized scores for each class
          in the predicted segmentation;
        - The loss used is a weighted combination of Dice and cross-entropy.
    """

    def __init__(self, image_tag: str, mask_tag: str, ce_weight: float = 0.1, dice_weight: float = 1, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            image_tag: Key to locate the input image from all the data returned in a batch.
            mask_tag: Key to locate the target segmentation mask from all the data returned in a batch.
            ce_weight: Weight to give to the cross-entropy factor of the segmentation loss
            dice_weight: Weight to give to the cross-entropy factor of the segmentation loss
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")
        self.model = self.configure_model()

    @auto_move_data
    def forward(self, *args, **kwargs):  # noqa: D102
        return self.model(*args, **kwargs)

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[self.hparams.image_tag], batch[self.hparams.mask_tag]

        # Forward
        y_hat = self.model(x)

        # Segmentation accuracy metrics
        ce = F.cross_entropy(y_hat, y)
        dice_values = self._dice(y_hat, y)
        dices = {f"dice/{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.ce_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:  # noqa: D102
        x = batch[self.hparams.image_tag]

        # Split the sequences in batches, in case the sequences are bigger than the batch size that fits in memory
        y_hat = []
        batch_size = self.trainer.datamodule.batch_size
        for batch_idx in range(int(np.ceil(len(x) / batch_size))):
            x_batch = x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            y_hat.append(self(x_batch))
        y_hat = torch.cat(y_hat)  # Assemble the segmentation of the whole batch from that of the sub-batches

        return y_hat


class RoiSegmentationTask(SegmentationTask):
    """Generic segmentation training steps for methods who perform multi-task ROI localization and segmentation.

    Implements generic segmentation train/val step, assuming the following conditions:
        - the model from ``self.configure_model()`` returns three outputs when specifying `predict=False`, namely i) a
          first rough segmentation from the whole image, ii) the bbox coordinates around the RoI, and iii) a refined
          segmentation of only the RoI.
        - The loss used is a weighted combination of Dice and cross-entropy + MAE on the regressed bbox coordinates.
    """

    def __init__(self, bbox_loss_weight: float = 20, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            bbox_loss_weight: Weight applied to the bbox prediction's MAE in the computation of the global loss.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self._roi_dice = DifferentiableDiceCoefficient(include_background=True, reduction="none")

    def _compute_normalized_bbox(self, y: Tensor) -> Tensor:
        """Computes the normalized coordinates of the bbox around the RoI in the segmentation.

        A normalized coordinate is mapped to the [0, 1] interval (to be able to be used in differentiable regression),
        rather than mapped to the [0, dim_size - 1] integer interval.

        Args:
            y: (N, H, W), Segmentation in categorical format.

        Returns:
            (N, 4), Normalized coordinates of the bbox around the RoI, in (x1, y1, x2, y2) format.
        """
        boxes = []
        for y_item in y:
            item_box = Measure.bbox(y_item, labels=range(1, len(self.hparams.data_params.labels)), normalize=True)
            # Convert between the (y1, x1, y2, x2) format from the `Measure` API and
            # the (x1, y1, x2, y2) format used by torchvision's RoI
            item_box = item_box[[1, 0, 3, 2]]
            boxes.append(item_box)
        return torch.stack(boxes).to(y.device)

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch[self.hparams.image_tag], batch[self.hparams.mask_tag]
        roi_bbox = self._compute_normalized_bbox(y)  # Compute the target RoI bbox from the groundtruth

        # Forward
        y_hat, roi_bbox_hat, roi_y_hat = self.model(x, predict=False)

        # Crop and resize ``y`` based on ``roi_bbox_hat`` predicted by the model
        boxes = torch.split(Measure.denormalize_bbox(roi_bbox_hat, self.hparams.data_params.out_shape[1:][::-1]), 1)
        roi_y = roi_pool(y.unsqueeze(1).to(boxes[0].dtype), boxes, roi_y_hat.shape[2:]).squeeze().long()

        # Global segmentation accuracy metrics
        ce = F.cross_entropy(y_hat, y)
        dice_values = self._dice(y_hat, y)
        dices = {f"global/dice/{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        # Localized segmentation accuracy metrics (includes the background since it is no longer dominant)
        roi_ce = F.cross_entropy(roi_y_hat, roi_y)
        roi_dice_values = self._roi_dice(roi_y_hat, roi_y)
        roi_dices = {f"dice/{label}": dice for label, dice in zip(self.hparams.data_params.labels, roi_dice_values)}
        roi_mean_dice = roi_dice_values.mean()

        # Regression metrics
        roi_bbox_mae = F.l1_loss(roi_bbox_hat, roi_bbox)

        loss = (
            (self.hparams.ce_weight * ce)
            + (self.hparams.dice_weight * (1 - mean_dice))
            + (self.hparams.ce_weight * roi_ce)
            + (self.hparams.dice_weight * (1 - roi_mean_dice))
            + (self.hparams.bbox_loss_weight * roi_bbox_mae)
        )

        # Format output
        return {
            "global/dice": mean_dice,
            **dices,
            "global/ce": ce,
            "dice": roi_mean_dice,
            **roi_dices,
            "ce": roi_ce,
            "roi_bbox_mae": roi_bbox_mae,
            "loss": loss,
        }
