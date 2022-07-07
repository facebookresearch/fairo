# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import polymetis_pb2
import polysim


class DoubleIntegratorSim:
    def __init__(self, hz, m=1.0):
        self.dt = 1.0 / hz
        self.m = m

        # Sim states
        self.x = 0.0
        self.v = 0.0
        self.f_buffer = 0.0

    def get_state(self) -> polymetis_pb2.RobotState:
        state = polymetis_pb2.RobotState()
        state.timestamp.GetCurrentTime()
        state.joint_positions[:] = [self.x]
        state.joint_velocities[:] = [self.v]

        return state

    def apply_control(self, cmd: polymetis_pb2.TorqueCommand):
        self.f_buffer = cmd.joint_torques[0]

    def step(self):
        acc = self.f_buffer / self.m
        self.x += self.v * self.dt + 0.5 * acc * self.dt ** 2
        self.v += acc * self.dt


if __name__ == "__main__":
    hz = 60
    mass = 1

    sim = DoubleIntegratorSim(hz, mass)

    polymetis_client = polysim.SimInterface(hz=hz)
    polymetis_client.register_arm_control(
        server_address="localhost:50051",
        state_callback=sim.get_state,
        action_callback=sim.apply_control,
        default_Kq=[1.0],
        default_Kqd=[0.1],
        dof=1,
    )
    polymetis_client.register_step_callback(sim.step)

    polymetis_client.run()
