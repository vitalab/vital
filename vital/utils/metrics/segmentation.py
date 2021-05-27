from typing import Callable, Dict, Sequence

import torch
from torch import Tensor

from vital.metrics.train.metric import DifferentiableDiceCoefficient


class DiceMetric:
    """Metric to compute dice score.

    Args:
        labels: list of names of the labels.
    """

    def __init__(self, labels: Sequence[str]):
        self.labels = labels
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")

    def __call__(self, output: Tensor, target: Tensor, **kwargs) -> Dict:
        """Compute dice.

        Args:
            output: Model prediction
            target: ground-truth
            **kwargs: extra arguments

        Returns:
            dice for all labels and average
        """
        with torch.no_grad():
            dice_values = self._dice(output, target)
            dices = {f"dice_{label}": dice for label, dice in zip(self.labels[1:], dice_values)}
            dices["dice"] = dice_values.mean()
            return dices
