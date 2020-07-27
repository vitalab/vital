import torch
from pytorch_lightning.metrics.functional.reduction import reduce
from torch import Tensor
from torch.nn import functional as F


def tversky_score(
    input: Tensor,
    target: Tensor,
    beta: float = 0.5,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: str = "elementwise_mean",
) -> Tensor:
    """Computes the loss definition of the Tversky index.

    The implementation of the score was inspired by PyTorch-Lightning's ``dice_score`` implementation (link in the
    refs), with the difference that the actual equation is a differentiable (i.e. `loss`) version of the score.

    References:
        - PyTorch-Lightning's ``dice_score`` implementation:
          https://pytorch-lightning.readthedocs.io/en/stable/metrics.html#dice-score-f
        - Description of the Tversky loss [accessed 22/06/2020]:
          https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html

    Args:
        input: (N, C, H, W), Raw, unnormalized scores for each class.
        target: (N, H, W), Groundtruth labels, where each value is 0 <= targets[i] <= C-1.
        beta: Weight to apply to false positives, and complement of the weight to apply to false negatives.
        bg: Whether to also compute dice_score for the background.
        nan_score: Score to return, if a NaN occurs during computation (denom zero).
        no_fg_score: Score to return, if no foreground pixel was found in target.
        reduction: Method for reducing accuracies over labels (default: takes the mean).
            Available reduction methods:
            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Returns:
        (1,) or (C,), the calculated Tversky index, average/sum or by labels.
    """
    n_classes = input.shape[1]
    bg = 1 - int(bool(bg))
    pred = F.softmax(input, dim=1)  # Use the softmax probability of the correct label instead of a hard label
    scores = torch.zeros(n_classes - bg, device=input.device, dtype=torch.float32)
    for i in range(bg, n_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg] += no_fg_score
            continue

        # Derivable version of the usual TP, FP and FN stats
        class_pred = pred[:, i, ...]
        tp = (class_pred * (target == i)).sum()
        fp = (class_pred * (target != i)).sum()
        fn = ((1 - class_pred) * (target == i)).sum()

        denom = tp + (beta * fp) + ((1 - beta) * fn)

        if torch.isclose(denom, torch.zeros_like(denom)).any():
            # nan result
            score_cls = nan_score
        else:
            score_cls = tp / denom

        scores[i - bg] += score_cls
    return reduce(scores, reduction=reduction)


def dice_score(
    input: Tensor,
    target: Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: str = "elementwise_mean",
) -> Tensor:
    """Computes the loss definition of the dice coefficient.

    Args:
        input: (N, C, H, W), Raw, unnormalized scores for each class.
        target: (N, H, W), Groundtruth labels, where each value is 0 <= targets[i] <= C-1.
        bg: Whether to also compute dice_score for the background.
        nan_score: Score to return, if a NaN occurs during computation (denom zero).
        no_fg_score: Score to return, if no foreground pixel was found in target.
        reduction: Method for reducing accuracies over labels (default: takes the mean).
            Available reduction methods:
            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Returns:
        (1,) or (C,), Calculated dice coefficient, average/sum or by labels.
    """
    return tversky_score(
        input, target, beta=0.5, bg=bg, nan_score=nan_score, no_fg_score=no_fg_score, reduction=reduction
    )


def kl_div_zmuv(mu: Tensor, logvar: Tensor, reduction: str = "elementwise_mean") -> Tensor:
    """Computes the KL divergence between a specified distribution and a N(0,1) Gaussian distribution.

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.

    Notes:
        - 'zmuv' stands for Zero Mean, Unit Variance.

    Args:
        mu: Mean of the distribution to compare to N(0,1).
        logvar: Log variance of the distribution to compare to N(0,1).
        reduction: String specifying the reduction method ('elementwise_mean', 'sum').

    Returns:
        (1,), KL divergence term of the VAE's loss.
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return reduce(kl_div, reduction=reduction)
