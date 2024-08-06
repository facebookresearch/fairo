# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import pytest

import grpc
import polymetis_pb2_grpc

from polysim import GrpcSimulationClient
from polysim.test_utils import fake_metadata_cfg, FakeEnv, FakeChannel, FakeConnection

N_DIM = 7
HZ = 250
STEPS = 100


def test_spinner(monkeypatch):
    # Patch grpc connection
    monkeypatch.setattr(grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(
        polymetis_pb2_grpc, "PolymetisControllerServerStub", FakeConnection
    )

    # Initialize env
    dt = 1.0 / HZ
    env = FakeEnv(N_DIM)
    fake_metadata_cfg.hz = HZ
    sim = GrpcSimulationClient(env=env, metadata_cfg=fake_metadata_cfg)

    # Run env
    t0 = time.time()
    sim.run(time_horizon=STEPS)
