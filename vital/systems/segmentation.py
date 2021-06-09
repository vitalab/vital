from typing import Dict

from torch import Tensor, nn
from torch.nn import functional as F

from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.systems.computation import TrainValComputationMixin
from vital.utils.decorators import auto_move_data


class SegmentationComputationMixin(TrainValComputationMixin):
    """Mixin for segmentation train/val step.

    Implements generic segmentation train/val step and inference, assuming the following conditions:
        - the ``nn.Module`` used returns as single output the raw, unnormalized scores for each class in the predicted
          segmentation.
    """

    # Fields to initialize in implementation of ``VitalSystem``
    #: Network called by ``SegmentationComputationMixin`` for test-time inference
    module: nn.Module

    def __init__(self, module: nn.Module, cross_entropy_weight: float = 0.1, dice_weight: float = 1, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            module: Module to train.
            cross_entropy_weight: Weight to give to the cross-entropy factor of the segmentation loss
            dice_weight: Weight to give to the cross-entropy factor of the segmentation loss
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")
        self.module = module
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight

    @auto_move_data
    def forward(self, *args, **kwargs):  # noqa: D102
        return self.module(*args, **kwargs)

    def trainval_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        y_hat = self.module(x)

        # Segmentation accuracy metrics
        ce = F.cross_entropy(y_hat, y)
        dice_values = self._dice(y_hat, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.trainer.datamodule.label_tags[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.cross_entropy_weight * ce) + (self.dice_weight * (1 - mean_dice))

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}
