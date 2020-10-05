from typing import Any

from pytorch_lightning.metrics import TensorMetric
from torch import Tensor

from vital.metrics.train.functionnal import differentiable_dice_score, kl_div_zmuv


class DifferentiableDiceCoefficient(TensorMetric):
    """Computes a differentiable version of the dice coefficient."""

    def __init__(
        self,
        include_background: bool = False,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = "elementwise_mean",
        reduce_group: Any = None,
    ):  # noqa: D205,D212,D415
        """
        Args:
            include_background: Whether to also compute dice for the background.
            nan_score: Score to return, if a NaN occurs during computation (denom zero).
            no_fg_score: Score to return, if no foreground pixel was found in target.
            reduction: Method for reducing metric score over labels.
                Available reduction methods:
                - ``'elementwise_mean'``: takes the mean (default)
                - ``'none'``: no reduction will be applied
            reduce_group: Process group to reduce metric results from DDP.
        """
        super().__init__(name="differentiable_dice", reduce_group=reduce_group)
        self.include_background = include_background
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        assert reduction in ("elementwise_mean", "none")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Actual metric calculation.

        Args:
            input: (N, C, H, W), Raw, unnormalized scores for each class.
            target: (N, H, W), Groundtruth labels, where each value is 0 <= targets[i] <= C-1.

        Return:
            (1,) or (C,), Calculated dice coefficient, averaged or by labels.
        """
        return differentiable_dice_score(
            input=input,
            target=target,
            bg=self.include_background,
            nan_score=self.nan_score,
            no_fg_score=self.no_fg_score,
            reduction=self.reduction,
        )


class KlDivergenceToZeroMeanUnitVariance(TensorMetric):
    """Computes the KL divergence between a specified distribution and a N(0,1) Gaussian distribution.

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.
    """

    def __init__(self, reduce_group: Any = None):  # noqa: D205,D212,D415
        """
        Args:
            reduce_group: Process group to reduce metric results from DDP.
        """
        super().__init__(name="kl_div_zmuv", reduce_group=reduce_group)

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Actual metric calculation.

        Args:
            mu: (N, Z), Mean of the distribution to compare to N(0,1).
            logvar: (N, Z) Log variance of the distribution to compare to N(0,1).

        Returns:
            (1,), KL divergence term of the VAE's loss.
        """
        return kl_div_zmuv(mu, logvar)
