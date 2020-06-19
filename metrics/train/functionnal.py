import torch
from pytorch_lightning.metrics.functional.reduction import reduce
from torch import Tensor
from torch.nn import functional as F


def dice_score(input: Tensor, target: Tensor, bg: bool = False,
               nan_score: float = 0.0, no_fg_score: float = 0.0,
               reduction: str = 'elementwise_mean') -> Tensor:
    """Computes a differentiable version of the dice_score coefficient.

    Inspired by Pytorch Lightning's ``dice_score`` implementation,
    only the TP, FP and FN were adapted to be differentiable.

    Args:
        input: (N, C, H, W), raw, unnormalized scores for each class.
        target: (N, H, W), where each value is 0 <= targets[i] <= C-1.
        bg: whether to also compute dice_score for the background.
        nan_score: score to return, if a NaN occurs during computation (denom zero).
        no_fg_score: score to return, if no foreground pixel was found in target.
        reduction: a method for reducing accuracies over labels (default: takes the mean)
            Available reduction methods:
            - elementwise_mean: takes the mean
            - none: pass array
            - sum: add elements

    Returns:
        (1,) or (C,), the calculated dice_score coefficient, average/summed or by labels.
    """
    n_classes = input.shape[1]
    bg = (1 - int(bool(bg)))
    pred = F.softmax(input, dim=1)  # Use the softmax probability of the correct label instead of a hard label
    scores = torch.zeros(n_classes - bg, device=input.device, dtype=torch.float32)
    for i in range(bg, n_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg] += no_fg_score
            continue

        # Derivable version of the usual TP, FP and FN stats
        tp = ((pred[:, i, ...]) * (target == i)).sum()
        fp = ((pred[:, i, ...]) * (target != i)).sum()
        pred_prob, pred_label = pred.max(dim=1)
        fn = ((pred_prob * (pred_label != i)) * (target == i)).sum()

        denom = (2 * tp + fp + fn)

        if torch.isclose(denom, torch.zeros_like(denom)).any():
            # nan result
            score_cls = nan_score
        else:
            score_cls = (2 * tp) / denom

        scores[i - bg] += score_cls
    return reduce(scores, reduction=reduction)


def kl_div_zmuv(mu: Tensor, logvar: Tensor, reduction: str = 'elementwise_mean') -> Tensor:
    """Computes the KL divergence between the distribution described by parameters ``mu`` and ``logvar``
    and a Zero Mean, Unit Variance (ZMUV) Gaussian distribution i.e. N(0,1).

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.

    Args:
        mu: mean of the distribution to compare to N(0,1).
        logvar: log variance of the distribution to compare to N(0,1).
        reduction: a string specifying the reduction method ('elementwise_mean', 'sum')

    Returns:
        (1,) the KL divergence term of the VAE's loss.
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return reduce(kl_div, reduction=reduction)
