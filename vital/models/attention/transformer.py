import warnings
from typing import Optional, Tuple, cast

import torch
from torch import Tensor, nn

from vital.models.attention.layers import BidirectionalMultimodalAttention, MultiheadAttention
from vital.models.layers import ModuleType, get_nn_module


class Transformer(nn.Module):
    """Transformer with extra features.

    The implementation was adapted from the implementation of Feature-Tokenizer transformer by the following paper:
    - "XTab: Cross-table Pretraining for Tabular Transformers", available online at: https://arxiv.org/abs/2305.06090
    """

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:  # noqa: D102
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    def __init__(
        self,
        *,
        d_token: int,
        n_bidirectional_blocks: int,
        n_self_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
    ) -> None:
        """Initializes class instance.

        Args:
            d_token: The dimensionality of the tokens.
            n_bidirectional_blocks: Number of the bidirectional multimodal attention blocks.
            n_self_blocks: Number of the self-attention blocks.
            attention_n_heads: Number of attention heads in each attention block.
            attention_dropout: Dropout ratio for the Multi Headed Attention module.
            attention_initialization: Weights initialization scheme for Multi Headed Attention module.
            attention_normalization: Normalization policy for attention layers. "layer_norm" is a good default.
            ffn_d_hidden: Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
            ffn_dropout: Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
            ffn_activation: Activation function type for the Feed-Forward Network module.
            ffn_normalization: Normalization scheme of the Feed-Forward Network module.
            residual_dropout: Dropout ratio for the output of the linear layers in attention block.
            prenormalization: Whether to apply normalization before the attention/linear layers, which typically
                stabilizes training.
            first_prenormalization: Whether to apply normalization in the first block, since the FTTransformer that
                inspired this code performs significantly worse with this option enabled. Only taken into account if
                `prenormalization` is True.
        """
        super().__init__()
        if not prenormalization:
            if first_prenormalization:
                raise ValueError("If `prenormalization` is False, then `first_prenormalization` must be False")

            if self.WARNINGS["prenormalization"]:
                cls_path = f"{self.__module__}.{self.__class__.__qualname__}"
                warnings.warn(
                    "`prenormalization` is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    f"You can turn off this warning by tweaking the {cls_path}.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            cls_path = f"{self.__module__}.{self.__class__.__qualname__}"
            warnings.warn(
                "`first_prenormalization` is set to True. Are you sure about this? "
                "The FTTransformer that inspired this code performs SIGNIFICANTLY worse with this option enabled. "
                f"You can turn off this warning by tweaking the {cls_path}.WARNINGS dictionary.",
                UserWarning,
            )

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization

        self.n_bidirectional_blocks = n_bidirectional_blocks
        self.n_self_blocks = n_self_blocks

        layers = []

        if n_bidirectional_blocks:
            layers += [
                self._init_bidirectional_attention_block(layer_idx) for layer_idx in range(n_bidirectional_blocks)
            ]

        if n_self_blocks:
            layers += [self._init_attention_block(layer_idx) for layer_idx in range(n_self_blocks)]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _init_bidirectional_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "bidirectional_attention": BidirectionalMultimodalAttention(
                    self.d_token,
                    self.attention_n_heads,
                    self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                )
            }
        )
        for modality_idx in (0, 1):
            layer.update(
                {
                    f"mod_{modality_idx}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"mod_{modality_idx}_ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"mod_{modality_idx}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"mod_{modality_idx}_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"mod_{modality_idx}_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _start_residual(self, block: nn.ModuleDict, layer_name: str, x: Tensor, stage="self"):
        match stage:
            case "bidirectional":
                assert layer_name in ["mod_0_attention", "mod_0_ffn", "mod_1_attention", "mod_1_ffn"]
            case "self":
                assert layer_name in ["attention", "ffn"]
            case _:
                assert False, "`stage` should be either 'bidirectional' or 'self."

        normalized_x = x
        if self.prenormalization:
            if f"{layer_name}_normalization" in block:  # Can't use `get` since it is not implemented for `ModuleDict`
                normalized_x = block[f"{layer_name}_normalization"](normalized_x)
        return normalized_x

    def _end_residual(self, block: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage="self"):
        match stage:
            case "bidirectional":
                assert layer_name in ["mod_0_attention", "mod_0_ffn", "mod_1_attention", "mod_1_ffn"]
            case "self":
                assert layer_name in ["attention", "ffn"]
            case _:
                assert False, "`stage` should be either 'bidirectional' or 'self."

        x_residual = block[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = block[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x1: Optional[Tensor] = None) -> Tensor | Tuple[Tensor, Tensor]:
        """Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x1: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        if self.n_bidirectional_blocks and x1 is None:
            raise ValueError(
                "`x1`, from which K and V are extracted, must be provided since the model includes bidirectional "
                "attention blocks."
            )

        if x.ndim != 3:
            raise ValueError("The input must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x1 is not None and x1.ndim != 3:
            raise ValueError("The second input must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x1 is not None and x.shape[-1] != x1.shape[-1]:
            raise ValueError(
                "The token dimensionality must be the same for both modalities to perform cross-attention, meaning "
                "that the last dimension of `x` and `x1` must be the same."
            )

        bidirectional_blocks = self.blocks[: self.n_bidirectional_blocks]
        self_blocks = self.blocks[self.n_bidirectional_blocks :]

        for block in bidirectional_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities
            x_residual = self._start_residual(block, "mod_0_attention", x, stage="bidirectional")
            x1_residual = self._start_residual(block, "mod_1_attention", x1, stage="bidirectional")

            # Forward pass through the bidirectional attention block
            x_residual, x1_residual = block["bidirectional_attention"](x_residual, x1_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "mod_0_attention", x, x_residual, stage="bidirectional")
            x1 = self._end_residual(block, "mod_1_attention", x1, x1_residual, stage="bidirectional")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "mod_0_ffn", x, stage="bidirectional")
            x_residual = block["mod_0_ffn"](x_residual)
            x = self._end_residual(block, "mod_0_ffn", x, x_residual, stage="bidirectional")

            x1_residual = self._start_residual(block, "mod_1_ffn", x1, stage="bidirectional")
            x1_residual = block["mod_1_ffn"](x1_residual)
            x1 = self._end_residual(block, "mod_1_ffn", x1, x1_residual, stage="bidirectional")

        if not self.n_self_blocks:
            # If there are no self-attention blocks, return the output tokens from both of the modalities
            return x, x1
        elif self.n_bidirectional_blocks:
            # If there were bidirectional attention blocks, concatenate the output from both modalities
            x = torch.cat([x, x1], dim=1)

        for block in self_blocks:
            block = cast(nn.ModuleDict, block)

            x_residual = self._start_residual(block, "attention", x)
            x_residual, _ = block["attention"](x_residual, x_residual)
            x = self._end_residual(block, "attention", x, x_residual)

            x_residual = self._start_residual(block, "ffn", x)
            x_residual = block["ffn"](x_residual)
            x = self._end_residual(block, "ffn", x, x_residual)

        return x
