from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from vital.models.layers import get_nn_module


class MLP(nn.Module):
    """Standard multilayer perceptron model.

    Args:
        input_shape: Shape of the input. If this does not specify a single dim, the input will be automatically
            flattened at the start of the forward pass.
        output_shape: Shape of the output. If this does not specify a single dim, the output will be automatically
            reshaped at the end of the forward pass.
        hidden: Number of neurons at each hidden layer of the MLP. If None or empty, the MLP will correspond to a linear
            model between the input and output.
        activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use in the MLP's
            hidden layers.
        output_activation: Name of the activation (as it is named in PyTorch's ``nn.Module`` package) to use at the
            MLP's last layer.
        dropout: Rate for dropout layers.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        hidden: Optional[Sequence[int]] = (128,),
        activation: str = "ReLU",
        output_activation: Optional[str] = None,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        input_dim = int(np.prod(input_shape))
        output_dim = int(np.prod(output_shape))

        if not hidden:
            hidden = []
        layers = [*hidden, output_dim]
        self.net = nn.Sequential()

        # Input layer
        self.net.add_module("layer_in", nn.Linear(input_dim, layers[0]))

        # Hidden layers
        for i in range(0, len(layers) - 1):
            self.net.add_module(f"{activation.lower()}_{i}", get_nn_module(activation))
            if dropout:
                self.net.add_module(f"drop_{i}", nn.Dropout(p=dropout))
            self.net.add_module(f"layer_{i+1}", nn.Linear(layers[i], layers[i + 1]))

        # Output layers
        if output_activation:
            self.net.add_module(f"{output_activation.lower()}_out", get_nn_module(output_activation))

    def forward(self, x: torch.Tensor):  # noqa D102
        x = torch.flatten(x, start_dim=1)
        x = self.net(x)
        x = torch.reshape(x, (-1, *self.output_shape))
        return x
