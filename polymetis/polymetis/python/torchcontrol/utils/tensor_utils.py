# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


def to_tensor(input):
    """Converts the input into a ``torch.Tensor`` of the default dtype."""
    if torch.is_tensor(input):
        return input.to(torch.Tensor())
    else:
        return torch.tensor(input).to(torch.Tensor())


def stack_trajectory(input):
    if torch.is_tensor(input):
        return input
    else:
        assert type(input) is list
        return torch.stack(input)


def diagonalize_gain(input: torch.Tensor):
    """Converts a 1-D vector into a diagonal 2-D matrix.

    - If the input tensor is 1-dimensional, interprets it as the diagonal
      and constructs the corresponding diagonal matrix.

    - If the input tensor is 2-dimensional, simply returns it.

    - Otherwise raises an AssertionError
    """
    assert (
        input.ndim == 1 or input.ndim == 2
    ), "Input gain has to be 1 or 2 dimensional tensor!"
    if input.ndim == 1:
        return torch.diag(input)
    else:
        return input
