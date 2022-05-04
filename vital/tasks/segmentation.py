from typing import Dict

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

    def __init__(self, ce_weight: float = 0.1, dice_weight: float = 1, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
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

    def _shared_train_val_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        y_hat = self.model(x)

        # Segmentation accuracy metrics
        ce = F.cross_entropy(y_hat, y)
        dice_values = self._dice(y_hat, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.ce_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}
