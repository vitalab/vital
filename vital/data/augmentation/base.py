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
        mask_token: (E) or (S, E) Mask to replace masked tokens with. If a single token of dimension (E), then the mask
            will be used to replace any tokens in the sequence. Otherwise, each token in the sequence has to to have its
            own MASK token to be replaced with.
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

    if mask_token.ndim == 1:
        mask_tokens = mask_token[None, None, ...].repeat(n, s, 1)
    elif mask_token.ndim == 2:
        mask_tokens = mask_token[None, ...].repeat(n, 1, 1)
    else:
        raise ValueError(
            "The `mask_token` parameter passed to `random_masking` should be of dimensions (E) or (S, E), where E is "
            "the embedding size and S is the sequence length."
        )

    x_masked = x * (1 - broadcast_mask) + mask_tokens * broadcast_mask
    return x_masked, mask
