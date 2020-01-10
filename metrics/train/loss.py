import torch
from torch import Tensor


def kl_div_zmuv(mu: Tensor, logvar: Tensor) -> Tensor:
    """ Computes the KL divergence between the distribution described by parameters ``mu`` and ``logvar``
    and a Zero Mean, Unit Variance (ZMUV) Gaussian distribution i.e. N(0,1).

    It is the standard loss to use for the reparametrization trick when training a variational autoencoder.

    Args:
        mu: mean of the distribution to compare to N(0,1).
        logvar: log variance of the distribution to compare to N(0,1).

    Returns:
        the KL divergence term of the VAE's loss.
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div
