import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vital.metrics.train.metric import DifferentiableDiceCoefficient


class DiceLoss(nn.Module):
    """Dice loss module."""

    def __init__(self):
        super().__init__()
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa D102
        dice_values = self._dice(input, target)
        mean_dice = dice_values.mean()
        return 1 - mean_dice


class DiceCELoss(nn.Module):
    """Dice and cross entropy loss  module."""

    def __init__(self, dice_weight=1, ce_weight=0.1):
        super().__init__()
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa D102
        ce = F.cross_entropy(input, target)
        dice_values = self._dice(input, target)
        mean_dice = dice_values.mean()
        loss = (self.ce_weight * ce) + (self.dice_weight * (1 - mean_dice))
        return loss
