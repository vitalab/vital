import math
from typing import Dict, Literal, Tuple

from torch import Tensor, nn
from torch.nn import functional as F


def reglu(x: Tensor) -> Tensor:
    """ReGLU activation function.

    References:
        - Noam Shazeer, "GLU Variants Improve Transformer", 2020, available online at: https://arxiv.org/abs/2002.05202
    """  # noqa: D403
    if x.shape[-1] % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2")
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """GEGLU activation function.

    References:
        - Noam Shazeer, "GLU Variants Improve Transformer", 2020, available online at: https://arxiv.org/abs/2002.05202
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2")
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """Class API for the ReGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return reglu(x)


class GEGLU(nn.Module):
    """Class API for the GEGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return geglu(x)


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' attention."""

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool = True,
        initialization: Literal["kaiming", "xavier"] = "kaiming"
    ) -> None:
        """Initializes class instance.

        Args:
            d_token: Token size. Must be a multiple of `n_heads`.
            n_heads: Number of attention heads. If greater than 1, then the module will have an additional output layer
                (so called "mixing" layer).
            dropout: Dropout rate for the attention map. The dropout is applied to *probabilities* and does not affect
                logits.
            bias: If `True`, then input (and output, if presented) layers also have bias.
            initialization: Initialization for input projection layers. Must be one of ['kaiming', 'xavier'].
        """
        super().__init__()
        if n_heads > 1:
            if d_token % n_heads != 0:
                raise ValueError("d_token must be a multiple of n_heads")

        if initialization not in ["kaiming", "xavier"]:
            raise ValueError("`initialization` must be one of ['kaiming', 'xavier']")

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == "xavier" and (m is not self.W_v or self.W_out is not None):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs a forward pass through the attention operations.

        Args:
            x_q: (N, S_q, E), query tokens.
            x_kv: (N, S_kv, E), key-value tokens.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            "attention_logits": attention_logits,
            "attention_probs": attention_probs,
        }
