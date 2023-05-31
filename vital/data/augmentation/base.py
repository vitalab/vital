from typing import Tuple

import torch
from torch import Tensor


def random_masking(x: Tensor, mask_token: Tensor, p: float) -> Tuple[Tensor, Tensor]:
    """Masks random tokens in sequences by replacing them with a predefined `mask_token`.

    References:
        - Adapted from the random masking implementation from the paper that introduced Mask Token Replacement (MTR):
          https://github.com/somaonishi/MTR/blob/33b87b37a63d120aff24c041da711fd8b714c00e/model/mask_token.py#L52-L68

    Args:
        x: (N, S, E) Batch of sequences of tokens.
        mask_token: (E) Mask to replace masked tokens with.
        p: Probability to replace a token by the mask token.

    Returns:
        x_masked: (N, S, E) Input tokens, where some tokens have been replaced by the mask token.
        mask: (N, S) Mask of tokens that were masked, with (1) representing tokens that were masked.
    """
    n, s, d = x.shape
    mask_dist = torch.full((n, s), p)  # Token-wise masking probability

    # Repeat the sampling in case all tokens are masked for an item in the batch
    mask = torch.bernoulli(mask_dist)
    while not mask.any(dim=1).all(dim=0):
        mask = torch.bernoulli(mask_dist)

    broadcast_mask = mask.unsqueeze(-1).to(device=x.device, dtype=torch.float)
    broadcast_mask = broadcast_mask.repeat(1, 1, d)

    mask_tokens = mask_token.repeat(n, s, 1)
    x_masked = x * (1 - broadcast_mask) + mask_tokens * broadcast_mask
    return x_masked, mask
