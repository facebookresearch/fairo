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


@pytest.mark.parametrize("hz, steps", [(60, 30), (250, 100)])
def test_spinner(monkeypatch, hz, steps):
    # Patch grpc connection
    monkeypatch.setattr(grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(
        polymetis_pb2_grpc, "PolymetisControllerServerStub", FakeConnection
    )

    # Initialize env
    dt = 1.0 / hz
    env = FakeEnv()
    fake_metadata_cfg.hz = hz
    sim = GrpcSimulationClient(env=env, metadata_cfg=fake_metadata_cfg)

    # Run env
    sim.run(time_horizon=5)  # warmup
    t0 = time.time()
    sim.run(time_horizon=steps)
    t1 = time.time()

    # Check sync clock time
    t_target = steps * dt
    t_actual = t1 - t0
    error = abs(t_target - t_actual) / steps / dt

    print("Testing async execution time...")
    print(f"Designated execution time: {t_target}")
    print(f"Actual execution time: {t_actual}")
    print(f"Percentage error per step: {error*100.0}%")

    assert error < 0.03
