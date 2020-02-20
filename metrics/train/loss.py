from typing import Literal

import torch
from torch import Tensor


def kl_div_zmuv(mu: Tensor, logvar: Tensor, reduction: Literal['mean', 'none'] = 'mean') -> Tensor:
    """Computes the KL divergence between the distribution described by parameters ``mu`` and ``logvar``
    and a Zero Mean, Unit Variance (ZMUV) Gaussian distribution i.e. N(0,1).

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.

    Args:
        mu: mean of the distribution to compare to N(0,1).
        logvar: log variance of the distribution to compare to N(0,1).
        reduction: specifies the reduction to apply to the output:
                   ``'none'``: no reduction will be applied,
                   ``'mean'``: the sum of the output will be divided by the number of elements in the output.

    Returns:
        the KL divergence term of the VAE's loss.
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if reduction == 'mean':
        kl_div = torch.mean(kl_div, dim=0)
    return kl_div
