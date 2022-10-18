import pytest

import grpc
import polymetis_pb2
import polymetis_pb2_grpc

import polysim
from polysim.test_utils import FakeEnv, FakeChannel, FakeConnection

N_DIM = 3
HZ = 250
STEPS = 100


class SimWrapper:
    def __init__(self):
        self.arm_env = FakeEnv(N_DIM)

    def arm_state_callback(self):
        return polymetis_pb2.RobotState()

    def arm_action_callback(self, cmd):
        pass

    def gripper_state_callback(self):
        return polymetis_pb2.GripperState()

    def gripper_action_callback(self, cmd):
        pass

    def step(self):
        pass


def test_sim_interface(monkeypatch):
    # Patch grpc connection
    monkeypatch.setattr(grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(
        polymetis_pb2_grpc, "PolymetisControllerServerStub", FakeConnection
    )

    # Initialize fake sim wrapper
    sim = SimWrapper()

    # Initialize sim client
    sim_client = polysim.SimInterface(hz=HZ)

    sim_client.register_arm_control(
        server_address="localhost:0000",
        state_callback=sim.arm_state_callback,
        action_callback=sim.arm_action_callback,
        dof=N_DIM,
    )
    sim_client.register_gripper_control(
        server_address=f"localhost:0000",
        state_callback=sim.gripper_state_callback,
        action_callback=sim.gripper_action_callback,
    )
    sim_client.register_step_callback(sim.step)

    # Start sim client
    sim_client.run(time_horizon=STEPS)
