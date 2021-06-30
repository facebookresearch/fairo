import torch


def timestamp_subtract(ts1, ts2):
    assert ts1.shape == torch.Size([2])
    assert ts2.shape == torch.Size([2])
    return ts1 - ts2


def timestamp_diff_seconds(ts1, ts2):
    ts_diff = timestamp_subtract(ts1, ts2).to(torch.float32)
    return ts_diff[0] + 1e-9 * ts_diff[1]


def timestamp_diff_ms(ts1, ts2):
    ts_diff = timestamp_subtract(ts1, ts2).to(torch.float32)
    return 1e3 * ts_diff[0] + 1e-6 * ts_diff[1]
