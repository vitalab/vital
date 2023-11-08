from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.utilities.distributed import reduce


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
          https://pytorch-lightning.readthedocs.io/en/stable/metrics.html#dice-score-func
        - Description of the Tversky loss [accessed 22/06/2020]:
          https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html

    Args:
        input: (N, C, H, W), Raw, unnormalized scores for each class.
        target: (N, H, W), Groundtruth labels, where each value is 0 <= targets[i] <= C-1.
        beta: Weight to apply to false positives, and complement of the weight to apply to false negatives.
        bg: Whether to also compute the dice score for the background.
        nan_score: Score to return, if a NaN occurs during computation (denom zero).
        no_fg_score: Score to return, if no foreground pixel was found in target.
        reduction: Method for reducing metric score over labels.
            Available reduction methods:
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'none'``: no reduction will be applied

    Returns:
        (1,) or (C,), the calculated Tversky index, averaged or by labels.
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

        # Differentiable version of the usual TP, FP and FN stats
        class_pred = pred[:, i, ...]
        tp = (class_pred * (target == i)).sum()
        fp = (class_pred * (target != i)).sum()
        fn = ((1 - class_pred) * (target == i)).sum()

        denom = tp + (beta * fp) + ((1 - beta) * fn)
        # nan result
        score_cls = tp / denom if torch.is_nonzero(denom) else nan_score

        scores[i - bg] += score_cls
    return reduce(scores, reduction=reduction)


def differentiable_dice_score(
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
        bg: Whether to also compute differentiable_dice_score for the background.
        nan_score: Score to return, if a NaN occurs during computation (denom zero).
        no_fg_score: Score to return, if no foreground pixel was found in target.
        reduction: Method for reducing metric score over labels.
            Available reduction methods:
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'none'``: no reduction will be applied

    Returns:
        (1,) or (C,), Calculated dice coefficient, averaged or by labels.
    """
    return tversky_score(
        input, target, beta=0.5, bg=bg, nan_score=nan_score, no_fg_score=no_fg_score, reduction=reduction
    )


def kl_div_zmuv(mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes the KL divergence between a specified distribution and a N(0,1) Gaussian distribution.

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.

    Notes:
        - 'zmuv' stands for Zero Mean, Unit Variance.

    Args:
        mu: (N, Z), Mean of the distribution to compare to N(0,1).
        logvar: (N, Z) Log variance of the distribution to compare to N(0,1).

    Returns:
        (1,), KL divergence term of the VAE's loss.
    """
    kl_div_by_samples = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return reduce(kl_div_by_samples, reduction="elementwise_mean")


def monotonic_regularization_loss(input: Tensor, target: Tensor, delta: float) -> Tensor:
    """Computes a regularization loss that enforces a monotonic relationship between the input and target.

    Notes:
        - This is a generalization of the attribute regularization loss proposed by the AR-VAE
          (link to the paper: https://arxiv.org/pdf/2004.05485.pdf)

    Args:
        input: (N, [1]), Input values to regularize so that they have a monotonic relationship with the `target` values.
        target: (N, [1]), Values used to determine the target monotonic ordering of the values.
        delta: Hyperparameter that decides the spread of the posterior distribution.

    Returns:
        (1,), Monotonic regularization term for aligning the input to the target.
    """
    # Compute input distance matrix
    broad_input = input.view(-1, 1).repeat(1, len(input))
    input_dist_mat = broad_input - broad_input.transpose(1, 0)

    # Compute target distance matrix
    broad_target = target.view(-1, 1).repeat(1, len(target))
    target_dist_mat = broad_target - broad_target.transpose(1, 0)

    # Compute regularization loss
    input_tanh = torch.tanh(input_dist_mat * delta)
    target_sign = torch.sign(target_dist_mat)
    loss = F.l1_loss(input_tanh, target_sign)

    return loss


def ntxent_loss(z_i: Tensor, z_j: Tensor, temperature: float = 1) -> Tensor:
    """Computes the NT-Xent loss for contrastive learning.

    References:
        - Inspired by the implementation from https://github.com/clabrugere/pytorch-scarf/blob/09281499d7c15dff3b1c49ba3d0957580e324641/scarf/loss.py#L6-L44

    Args:
        z_i: (N, E), Embedding of one view of the input data.
        z_j: (N, E), Embedding of the other view of the input data.
        temperature: Scaling factor of the similarity metric.

    Returns:
        (1,), Calculated NT-Xent loss.
    """
    batch_size = z_i.size(0)

    # compute similarity between the embeddings of both views of the data
    z = torch.cat([z_i, z_j], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (N * 2, N * 2)

    # Create a mask for the positive samples, i.e. the corresponding samples in each view
    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives_mask = torch.cat([sim_ij, sim_ji], dim=0)

    # For the denominator, we need to include both the positive and negative samples, so we use the inverse of the
    # identity matrix, so that we only exclude the similarities between each embedding and itself
    pairwise_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()

    numerator = torch.exp(positives_mask / temperature)
    denominator = pairwise_mask * torch.exp(similarity / temperature)

    all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
    loss = torch.sum(all_losses) / (2 * batch_size)

    return loss


def cdist(x1: Tensor, x2: Tensor, **kwargs) -> Tensor:
    """Wrapper around torch's native `cdist` function to use it on non-batched inputs.

    Args:
        x1: ([B,]P,M), Input collection of row tensors.
        x2: ([B,]R,M), Input collection of row tensors.
        **kwargs: Additional parameters to pass along to torch's native `cdist`.

    Returns:
        ([B,]P,R), Pairwise p-norm distances between the row tensors.
    """
    if x1.ndim != x2.ndim:
        raise ValueError(
            f"Wrapper around torch's `cdist` only supports when both input tensors are identically batched or not. "
            f"However, the current shapes do not match: {x1.shape=} and {x2.shape=}."
        )

    if no_batch := x1.ndim < 3:
        x1 = x1[None, ...]
        x2 = x2[None, ...]

    dist = torch.cdist(x1, x2, **kwargs)
    if no_batch:
        dist = dist[0]

    return dist


def rbf_kernel(x1: Tensor, x2: Tensor = None, length_scale: float | Sequence[float] | Tensor = 1) -> Tensor:
    """Computes the Radial Basis Function kernel (aka Gaussian kernel).

    Args:
        x1: (M, E), Left argument of the returned kernel k(x1,x2).
        x2: (N, E), Right argument of the returned kernel k(x1,x2). If None, uses `x2=x1`.
        length_scale: (1,) or (E,), The length-scale of the kernel. If a float, an isotropic kernel is used.
            If a Sequence or Tensor, an anisotropic kernel is used to define the length-scale of each feature dimension.
            If None, use Silverman's rule-of-thumb to compute an estimate of the optimal (anisotropic) length-scale.

    Returns:
        (M, N), The kernel k(x1,x2).
    """
    if isinstance(length_scale, Sequence):
        # Make sure that if the length-scale is specified for each feature dimension, it is in tensor format
        length_scale = torch.tensor(length_scale, device=x1.device)

    if x2 is None:
        x2 = x1.clone()

    # Use the trick of distributing `length_scale` on the inputs to minimize computations
    x1, x2 = x1 / length_scale, x2 / length_scale
    # Reshape inputs so that the squared Euclidean distance can be easily computed using simple broadcasting
    x1 = x1[:, None, :]  # (N, D) -> (N, 1, D)
    x2 = x2[None, :, :]  # (N, D) -> (1, N, D)

    # Compute the RBF kernel
    sq_dist = torch.sum((x1 - x2) ** 2, -1)  # sq_dist = |x1 - x2|^2 / l^2
    k = torch.exp(-0.5 * sq_dist)  # exp( - sq_dist / 2 )
    return k
