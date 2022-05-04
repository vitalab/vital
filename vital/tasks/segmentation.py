from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.tasks.generic import SharedTrainEvalTask
from vital.utils.decorators import auto_move_data


class SegmentationTask(SharedTrainEvalTask):
    """Generic segmentation training and inference steps.

    Implements generic segmentation train/val step and inference, assuming the following conditions:
        - the model from ``self.configure_model()`` returns as lone output the raw, unnormalized scores for each class
          in the predicted segmentation;
        - The loss used is a weighted combination of Dice and cross-entropy.
    """

    def __init__(self, cross_entropy_weight: float = 0.1, dice_weight: float = 1, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            cross_entropy_weight: Weight to give to the cross-entropy factor of the segmentation loss
            dice_weight: Weight to give to the cross-entropy factor of the segmentation loss
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")
        self.model = self.configure_model()
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight

    @auto_move_data
    def forward(self, *args, **kwargs):  # noqa: D102
        return self.model(*args, **kwargs)

    def _shared_train_val_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        y_hat = self.model(x)

        # Segmentation accuracy metrics
        ce = F.cross_entropy(y_hat, y)
        dice_values = self._dice(y_hat, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.cross_entropy_weight * ce) + (self.dice_weight * (1 - mean_dice))

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:  # noqa: D102
        x = batch[Tags.img]

        # Split the sequences in batches, in case the sequences are bigger than the batch size that fits in memory
        y_hat = []
        batch_size = self.trainer.datamodule.batch_size
        for batch_idx in range(int(np.ceil(len(x) / batch_size))):
            x_batch = x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            y_hat.append(self(x_batch))
        y_hat = torch.cat(y_hat)  # Assemble the segmentation of the whole batch from that of the sub-batches

        return y_hat
