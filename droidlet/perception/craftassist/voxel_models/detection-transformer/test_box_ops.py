"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import torch

import box_ops


class Tester(unittest.TestCase):
    def test_box_cxcywh_to_xyxy(self):
        t = torch.rand(10, 4)
        r = box_ops.box_xyxy_to_cxcywh(box_ops.box_cxcywh_to_xyxy(t))
        self.assertTrue((t - r).abs().max() < 1e-5)


if __name__ == "__main__":
    unittest.main()
