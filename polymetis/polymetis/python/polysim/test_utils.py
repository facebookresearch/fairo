# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for unit testing."""
from omegaconf.dictconfig import DictConfig
from polymetis.robot_client.metadata import RobotClientMetadata


fake_metadata_cfg = DictConfig(
    {
        "_target_": "polysim.test_utils.FakeMetadata",
        "hz": 10,
    }
)
"""
A fake metadata DictConfig for testing.
"""


class FakeMetadata(RobotClientMetadata):
    def __init__(self, hz):
        self.fake_dict = DictConfig({"hz": hz})

    def get_proto(self):
        return self.fake_dict


"""
A fake metadata for testing.
"""
