from typing import Any, Optional

import torch
from pytorch_lightning.metrics import Metric

from vital.metrics.train.functional import differentiable_dice_score, kl_div_zmuv


class DifferentiableDiceCoefficient(Metric):
    """Computes a differentiable version of the dice coefficient."""

    def __init__(
        self,
        include_background: bool = False,
        num_classes: int = None,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = "elementwise_mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):  # noqa: D205,D212,D415
        """
        Args:
            include_background: Whether to also compute dice for the background.
            num_classes: Number of classes the metric should return, if no reduction are applied.
                You only need to provide this argument when `reduction=='none'`.
            nan_score: Score to return, if a NaN occurs during computation (denom zero).
            no_fg_score: Score to return, if no foreground pixel was found in target.
            reduction: Method for reducing metric score over labels.
                Available reduction methods:
                - ``'elementwise_mean'``: takes the mean (default)
                - ``'none'``: no reduction will be applied
            compute_on_step: Forward only calls ``update()`` and returns None if this is set to False.
            dist_sync_on_step: Synchronize metric state across processes at each ``forward()`` before returning the
                value at the step.
            process_group: Specify the process group on which synchronization is called.
                Selects the entire world by default.
        """
        if reduction == "none" and num_classes is None:
            raise ValueError(
                "If you don't apply any reduction, you must provide the number of classes the metric should return. "
                "This number should take into account whether you also want to include the background or not."
            )

        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group
        )
        self.include_background = include_background
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        assert reduction in ("elementwise_mean", "none")
        self.reduction = reduction

        dice_state_default = torch.zeros(num_classes) if self.reduction == "none" else torch.tensor(0.0)
        self.add_state("sum_dice_score", dice_state_default, dist_reduce_fx="sum")
        self.add_state("counter", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with dice over the batch.

        Args:
            input: (N, C, H, W), Raw, unnormalized scores for each class.
            target: (N, H, W), Groundtruth labels, where each value is 0 <= targets[i] <= C-1.
        """
        self.sum_dice_score += differentiable_dice_score(
            input=input,
            target=target,
            bg=self.include_background,
            nan_score=self.nan_score,
            no_fg_score=self.no_fg_score,
            reduction=self.reduction,
        )
        self.counter += 1

    def compute(self) -> torch.Tensor:
        """Reduces dice over state.

        Return:
            (1,) or (C,), Calculated dice coefficient, averaged or by labels.
        """
        return self.sum_dice_score / self.counter


class KlDivergenceToZeroMeanUnitVariance(Metric):
    """Computes the KL divergence between a specified distribution and a N(0,1) Gaussian distribution.

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.
    """

    def __init__(
        self, compute_on_step: bool = True, dist_sync_on_step: bool = False, process_group: Optional[Any] = None
    ):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group
        )
        self.add_state("sum_kl_div", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counter", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mu: torch.Tensor, logvar: torch.Tensor) -> None:
        """Update state with KL divergence for the batch.

        Args:
            mu: (N, Z), Mean of the distribution to compare to N(0,1).
            logvar: (N, Z) Log variance of the distribution to compare to N(0,1).
        """
        self.sum_kl_div += kl_div_zmuv(mu, logvar)
        self.counter += 1

    def compute(self) -> torch.Tensor:
        """Reduces KL divergence over state.

        Returns:
            (1,), KL divergence term of the VAE's loss.
        """
        return self.sum_kl_div / self.counter
