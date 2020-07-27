from torch import Tensor, nn

from vital.metrics.train.functionnal import dice_score


class DiceCoefficient(nn.Module):
    """Computes a differentiable version of the dice coefficient."""

    def __init__(
        self,
        include_background: bool = False,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = "elementwise_mean",
    ):  # noqa: D205,D212,D415
        """
        Args:
            include_background: Whether to also compute dice for the background.
            nan_score: Score to return, if a NaN occurs during computation (denom zero).
            no_fg_score: Score to return, if no foreground pixel was found in target.
            reduction: Method for reducing accuracies over labels (default: takes the mean).
                Available reduction methods:
                - elementwise_mean: takes the mean
                - none: pass array
                - sum: add elements
        """
        super().__init__()
        self.include_background = include_background
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Actual metric computation.

        Args:
            input: (N, C, H, W), Raw, unnormalized scores for each class.
            target: (N, H, W), Groundtruth labels, where each value is 0 <= targets[i] <= C-1.

        Return:
            (1,) or (C,), Calculated dice coefficient, average/sum or by labels.
        """
        return dice_score(
            input=input,
            target=target,
            bg=self.include_background,
            nan_score=self.nan_score,
            no_fg_score=self.no_fg_score,
            reduction=self.reduction,
        )
