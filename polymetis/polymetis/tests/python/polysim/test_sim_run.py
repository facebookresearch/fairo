# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import pytest

import grpc
import polymetis_pb2
import polymetis_pb2_grpc

import numpy as np

from polysim import GrpcSimulationClient
from polysim.envs import AbstractControlledEnv
from polysim.test_utils import fake_metadata_cfg

N_DIM = 7
HZ = 250
STEPS = 100


class FakeEnv(AbstractControlledEnv):
    def reset(self):
        pass

    def get_num_dofs(self):
        pass

    def get_current_joint_pos_vel(self):
        return np.zeros(N_DIM), np.zeros(N_DIM)

    def get_current_joint_torques(self):
        return np.zeros(N_DIM), np.zeros(N_DIM), np.zeros(N_DIM), np.zeros(N_DIM)

    def apply_joint_torques(self, torques):
        pass


class FakeChannel:
    def __init__(self, ip):
        pass

    def close(self):
        pass


class FakeConnection:
    def __init__(self, channel):
        pass

    def ControlUpdate(self, robot_state):
        return polymetis_pb2.TorqueCommand()

    def InitRobotClient(self, metadata):
        pass


def test_spinner(monkeypatch):
    # Patch grpc connection
    monkeypatch.setattr(grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(
        polymetis_pb2_grpc, "PolymetisControllerServerStub", FakeConnection
    )

    # Initialize env
    dt = 1.0 / HZ
    env = FakeEnv()
    fake_metadata_cfg.hz = HZ
    sim = GrpcSimulationClient(env=env, metadata_cfg=fake_metadata_cfg)

    # Run env
    t0 = time.time()
    sim.run(time_horizon=STEPS)
