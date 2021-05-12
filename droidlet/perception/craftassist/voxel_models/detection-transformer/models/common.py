"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple feed forward fully connected, with some options

    Parameters
    ----------
    input_dim : int
        Number of input channels
    hidden_dim : int
        Number of channels in the hidden layers
    output_dim : int
        Number of output channels
    nb_layers : int
        Number of layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim, nb_layers=1):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(nb_layers):
            is_last_layer = i == nb_layers - 1
            cur_in = input_dim if i == 0 else hidden_dim
            cur_out = output_dim if is_last_layer else hidden_dim
            linear = nn.Linear(cur_in, cur_out)
            self.layers.append(linear)

    def forward(self, x):  # pylint: disable=arguments-differ
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
