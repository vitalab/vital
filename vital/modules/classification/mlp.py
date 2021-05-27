from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Standard Multi-layer Perceptron model.

    Args:
        input_dim: number of input neurons
        hidden: tuple of number of hidden neurons
        output_dim: number of output neurons
        output_activation: activation function for last layer
        dropout_rate: rate for dropout layers
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        ouput_shape: Tuple[int],
        hidden: Sequence[int] = (128,),
        output_activation: Optional[nn.Module] = None,
        dropout_rate: float = 0.25,
    ):

        input_dim = int(np.prod(input_shape))
        assert len(ouput_shape) == 1, "Output shape must be 1 dimension"
        output_dim = int(ouput_shape[0])

        super(MLP, self).__init__()

        self.net = nn.Sequential()

        # Input layer
        self.net.add_module("input_layer", nn.Linear(input_dim, hidden[0]))
        self.net.add_module("relu_{}".format(0), nn.ReLU())
        self.net.add_module("drop_{}".format(0), nn.Dropout(p=dropout_rate))

        # Hidden layers
        for i in range(0, len(hidden) - 1):
            self.net.add_module("layer_{}".format(i + 1), nn.Linear(hidden[i], hidden[i + 1]))
            self.net.add_module("relu_{}".format(i + 1), nn.ReLU())
            self.net.add_module("drop_{}".format(i + 1), nn.Dropout(p=dropout_rate))

        # Output layers
        self.net.add_module("output_layer", nn.Linear(hidden[-1], output_dim))
        if output_activation:
            self.net.add_module("output_activation", output_activation)

    def forward(self, x: torch.Tensor):  # noqa D102
        x = torch.flatten(x, 1)
        return self.net(x)
