from typing import Any

from pytorch_lightning.metrics import TensorMetric
from torch import Tensor

from vital.metrics.train.functionnal import dice_score


class DiceCoefficient(TensorMetric):
    """Computes a differentiable version of the dice_score coefficient."""

    def __init__(self, include_background: bool = False,
                 nan_score: float = 0.0, no_fg_score: float = 0.0,
                 reduction: str = 'elementwise_mean',
                 reduce_group: Any = None, reduce_op: Any = None):
        """
        Args:
            include_background: whether to also compute dice_score for the background.
            nan_score: score to return, if a NaN occurs during computation (denom zero).
            no_fg_score: score to return, if no foreground pixel was found in target.
            reduction: a method for reducing accuracies over labels (default: takes the mean)
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__(name='dice', reduce_group=reduce_group, reduce_op=reduce_op)

        self.include_background = include_background
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Actual metric computation.

        Args:
            input: (N, C, H, W), raw, unnormalized scores for each class.
            target: (N, H, W), where each value is 0 <= targets[i] <= C-1.

        Return:
            (1,) or (C,), the calculated dice_score coefficient, average/summed or by labels.
        """
        return dice_score(input=input, target=target, bg=self.include_background,
                          nan_score=self.nan_score, no_fg_score=self.no_fg_score,
                          reduction=self.reduction)
