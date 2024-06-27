import math
from typing import Dict, Literal, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init


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


class _QKVLinearProjection(nn.Module):
    def __init__(
        self, d_token: int, n_heads: int, bias: bool = True, initialization: Literal["kaiming", "xavier"] = "kaiming"
    ):
        """Initializes class instance.

        Args:
            d_token: Token size.
            n_heads: Number of attention heads. If equal to 1, then value projection matrix will always be initialized
                with Kaiming (regardless of `initialization` parameter), to follow torch.nn.MultiheadAttention.
            bias: If `True`, then input (and output, if presented) layers also have bias.
            initialization: Initialization for input projection layers. Must be one of ['kaiming', 'xavier'].
        """
        super().__init__()

        if initialization not in ["kaiming", "xavier"]:
            raise ValueError("`initialization` must be one of ['kaiming', 'xavier']")

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if V is directly used to compute output (i.e. not multi-head);
            # the latter one is initialized with Kaiming in torch
            if initialization == "xavier" and (m is not self.W_v or n_heads > 1):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes the query/key/value linear projections of tokens.

        Args:
            x_q: (N, S_q, E), Tokens from which to compute the query matrix.
            x_kv: (N, S_kv, E), Tokens from which to compute the key/value matrices.

        Returns:
            (N, S_q, E) + 2 x (N, S_kv, E), query/key/value linear projections of input tokens.
        """
        return self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)


class _QKVMatrixMultiplication(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float, bias: bool = True):
        """Initializes class instance.

        Args:
            d_token: Token size. Must be a multiple of `n_heads`.
            n_heads: Number of attention heads. If greater than 1, then the module will have an additional output layer
                (so called "mixing" layer).
            dropout: Dropout rate for the attention map. The dropout is applied to *probabilities* and does not affect
                logits.
            bias: If `True`, then input (and output, if presented) layers also have bias.
        """
        super().__init__()

        if n_heads > 1:
            if d_token % n_heads != 0:
                raise ValueError("d_token must be a multiple of n_heads")

        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

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

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs the multiplications between query/key/value matrices.

        Args:
            q: (N, S_q, E), query matrix.
            k: (N, S_kv, E), key matrix.
            v: (N, S_kv, E), value matrix.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
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


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-)."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool = True,
        initialization: Literal["kaiming", "xavier"] = "kaiming",
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
        self.linear_proj = _QKVLinearProjection(d_token, n_heads, bias=bias, initialization=initialization)
        self.mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs a forward pass through the attention operations.

        Args:
            x_q: (N, S_q, E), query tokens.
            x_kv: (N, S_kv, E), key-value tokens.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        q, k, v = self.linear_proj(x_q, x_kv)
        return self.mat_mul(q, k, v)


class PositionalEncoding(nn.Module):
    """Positional encoding layer."""

    def __init__(self, sequence_len: int, d_model: int):
        """Initializes layers parameters.

        Args:
            sequence_len: The number of tokens in the input sequence.
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        self.positional_encoding = Parameter(torch.empty(sequence_len, d_model))
        init.trunc_normal_(self.positional_encoding, std=0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that adds positional encoding to the input tensor.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, S, `d_model`), Tensor with added positional encoding.
        """
        return x + self.positional_encoding[None, ...]


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    When used as a module, the [CLS]-token is appended **to the end** of each item in the batch.

    Notes:
        - This is a port of the `CLSToken` class from v0.0.13 of the `rtdl` package. It mixes the original
          implementation with the simpler code of `_CLSEmbedding` from v0.0.2 of the `rtdl_revisiting_models` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L380-L446

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d_token = 4
            cls_token = CLSToken(d_token, 'uniform')
            x = torch.randn(batch_size, n_tokens, d_token)
            x = cls_token(x)
            assert x.shape == (batch_size, n_tokens + 1, d_token)
            assert (x[:, -1, :] == cls_token.expand(len(x))).all()
    """

    def __init__(self, d_token: int) -> None:
        """Initializes class instance.

        Args:
            d_token: the size of token
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the weights using a uniform distribution."""
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the underlying :code:`weight` parameter, so
            gradients will be propagated as expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class SequencePooling(nn.Module):
    """Sequence pooling layer."""

    def __init__(self, d_model: int):
        """Initializes layer submodules.

        Args:
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        # Initialize the learnable parameters of the sequential pooling
        self.attention_pool = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that performs a (learnable) weighted averaging of the different tokens.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, `d_model`), Output tensor.
        """
        attn_vector = F.softmax(self.attention_pool(x), dim=1)  # (N, S, 1)
        broadcast_attn_vector = attn_vector.transpose(2, 1)  # (N, S, 1) -> (N, 1, S)
        pooled_x = (broadcast_attn_vector @ x).squeeze(1)  # (N, 1, S) @ (N, S, E) -> (N, E)
        return pooled_x
