import torch


def compute_quat_dist(a, b):
    return torch.acos((2 * (a * b).sum() ** 2 - 1).clip(-1, 1))
