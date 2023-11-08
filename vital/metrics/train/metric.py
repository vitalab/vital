from typing import Sequence

from torch import Tensor, nn

from vital.metrics.train.functional import (
    differentiable_dice_score,
    monotonic_regularization_loss,
    ntxent_loss,
    rbf_kernel,
)


class DifferentiableDiceCoefficient(nn.Module):
    """Computes a differentiable version of the dice coefficient."""

    def __init__(
        self,
        include_background: bool = False,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: str = "elementwise_mean",
    ):
        """Initializes class instance.

        Args:
            include_background: Whether to also compute dice for the background.
            nan_score: Score to return, if a NaN occurs during computation (denom zero).
            no_fg_score: Score to return, if no foreground pixel was found in target.
            reduction: Method for reducing metric score over labels.
                Available reduction methods:
                - ``'elementwise_mean'``: takes the mean (default)
                - ``'none'``: no reduction will be applied
        """
        super().__init__()
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


class MonotonicRegularizationLoss(nn.Module):
    """Computes a regularization loss that enforces a monotonic relationship between the input and target.

    Notes:
        - This is a generalization of the attribute regularization loss proposed by the AR-VAE
          (link to the paper: https://arxiv.org/pdf/2004.05485.pdf)
    """

    def __init__(self, delta: float):
        """Initializes class instance.

        Args:
            delta: Hyperparameter that decides the spread of the posterior distribution.
        """
        super().__init__()
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Actual metric calculation.

        Args:
            input: (N, [1]), Input values to regularize so that they have a monotonic relationship with the `target`
                values.
            target: (N, [1]), Values used to determine the target monotonic ordering of the values.

        Returns:
            (1,), Calculated monotonic regularization loss.
        """
        return monotonic_regularization_loss(input, target, self.delta)


class NTXent(nn.Module):
    """Computes the NT-Xent loss for contrastive learning."""

    def __init__(self, temperature=1.0):
        """Initializes class instance.

        Args:
            temperature: Scaling factor of the similarity metric.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor):
        """Actual metric calculation.

        Args:
            z_i: (N, E), Embedding of one view of the input data.
            z_j: (N, E), Embedding of the other view of the input data.

        Return:
            (1,), Calculated NT-Xent loss.
        """
        return ntxent_loss(z_i, z_j, temperature=self.temperature)


class RBFKernel(nn.Module):
    """Computes the Radial Basis Function kernel (aka Gaussian kernel)."""

    def __init__(self, length_scale: float | Sequence[float] | Tensor = 1):
        """Initializes class instance.

        Args:
            length_scale: length_scale: (1,) or (E,), The length-scale of the kernel. If a float, an isotropic kernel is
                used. If a Sequence or Tensor, an anisotropic kernel is used to define the length-scale of each feature
                dimension.
        """
        super().__init__()
        self.length_scale = length_scale

    def forward(self, x1: Tensor, x2: Tensor = None) -> Tensor:
        """Actual kernel calculation.

        Args:
            x1: (M, E), Left argument of the returned kernel k(x1,x2).
            x2: (N, E), Right argument of the returned kernel k(x1,x2). If None, uses `x2=x1`,
                which ends up evaluating k(x1,x1).

        Returns:
            (N, M), The kernel k(x1,x2).
        """
        return rbf_kernel(x1, x2=x2, length_scale=self.length_scale)
