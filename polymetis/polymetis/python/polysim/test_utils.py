# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for unit testing."""
from unittest.mock import MagicMock
from omegaconf.dictconfig import DictConfig

import numpy as np

import polymetis_pb2
from polymetis.robot_client.metadata import RobotClientMetadata
from polysim.envs import AbstractControlledEnv


"""
A fake metadata DictConfig for testing.
"""
fake_metadata_cfg = DictConfig(
    {
        "_target_": "polysim.test_utils.FakeMetadata",
        "hz": 10,
    }
)

"""
A fake metadata for testing.
"""


class FakeMetadata(RobotClientMetadata):
    def __init__(self, hz):
        self.fake_dict = DictConfig({"hz": hz})

    def get_proto(self):
        return self.fake_dict


"""
A fake simulation env for testing.
"""


class FakeEnv(AbstractControlledEnv):
    def __init__(self, n_dim=1):
        self.n_dim = n_dim

    def reset(self):
        pass

    def get_num_dofs(self):
        pass

    def get_current_joint_pos_vel(self):
        return np.zeros(self.n_dim), np.zeros(self.n_dim)

    def get_current_joint_torques(self):
        return (
            np.zeros(self.n_dim),
            np.zeros(self.n_dim),
            np.zeros(self.n_dim),
            np.zeros(self.n_dim),
        )

    def apply_joint_torques(self, torques):
        pass

    def set_robot_state(self, robot_state):
        pass


"""
Fake gRPC artifacts for testing.
Use by:
    monkeypatch.setattr(grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(
        polymetis_pb2_grpc, "PolymetisControllerServerStub", FakeConnection
    )
"""
FakeChannel = MagicMock()


class FakeConnection:
    def __init__(self, channel):
        pass

    def ControlUpdate(self, robot_state):
        return polymetis_pb2.TorqueCommand()

    def InitRobotClient(self, metadata):
        pass
