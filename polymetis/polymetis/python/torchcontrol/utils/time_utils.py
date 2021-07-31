# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


def timestamp_subtract(ts1, ts2):
    """Subtracts two timestamps

    Timestamps are represented as a torch.Tensor of shape (2,), where the first element
    corresponds to seconds and the second element corresponds to nanoseconds.

    Args:
        ts1: First timestamp
        ts2: Second timestamp

    Outputs:
        Result of ts1 - ts2 represented as a timestamp
    """

    assert ts1.shape == torch.Size([2])
    assert ts2.shape == torch.Size([2])
    return ts1 - ts2


def timestamp_diff_seconds(ts1, ts2):
    """Computes the time difference in seconds between two timestamps.

    Timestamps are represented as a torch.Tensor of shape (2,), where the first element
    corresponds to seconds and the second element corresponds to nanoseconds.

    Args:
        ts1: First timestamp
        ts2: Second timestamp

    Outputs:
        Result of ts1 - ts2 in seconds
    """
    ts_diff = timestamp_subtract(ts1, ts2).to(torch.float32)
    return ts_diff[0] + 1e-9 * ts_diff[1]


def timestamp_diff_ms(ts1, ts2):
    """Computes the time difference in milliseconds between two timestamps.

    Timestamps are represented as a torch.Tensor of shape (2,), where the first element
    corresponds to seconds and the second element corresponds to nanoseconds.

    Args:
        ts1: First timestamp
        ts2: Second timestamp

    Outputs:
        Result of ts1 - ts2 in milliseconds
    """
    ts_diff = timestamp_subtract(ts1, ts2).to(torch.float32)
    return 1e3 * ts_diff[0] + 1e-6 * ts_diff[1]
