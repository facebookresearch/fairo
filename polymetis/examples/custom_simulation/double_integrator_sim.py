# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import polymetis_pb2
import polysim


class DoubleIntegratorSim:
    def __init__(self, dof, hz, m=1.0):
        self.dof = dof
        self.dt = 1.0 / hz
        self.m = m

        # Sim states
        self.x = np.zeros(dof)
        self.v = np.zeros(dof)
        self.f_buffer = np.zeros(dof)
        self.t = 0

    def get_state(self) -> polymetis_pb2.RobotState:
        state = polymetis_pb2.RobotState()
        state.timestamp.FromNanoseconds(int(1e9 * self.t))
        state.joint_positions[:] = self.x
        state.joint_velocities[:] = self.v

        state.joint_torques_computed[:] = [0.0] * self.dof
        state.prev_joint_torques_computed[:] = [0.0] * self.dof
        state.prev_joint_torques_computed_safened[:] = [0.0] * self.dof
        state.motor_torques_measured[:] = [0.0] * self.dof
        state.motor_torques_external[:] = [0.0] * self.dof
        state.motor_torques_desired[:] = [0.0] * self.dof

        return state

    def apply_control(self, cmd: polymetis_pb2.TorqueCommand):
        assert len(cmd.joint_torques) == self.dof
        self.f_buffer = np.array(cmd.joint_torques)

    def step(self):
        acc = self.f_buffer / self.m
        self.x += self.v * self.dt + 0.5 * acc * self.dt**2
        self.v += acc * self.dt

        self.t += self.dt


if __name__ == "__main__":
    dof = 2
    hz = 60
    mass = 1

    sim = DoubleIntegratorSim(dof, hz, mass)

    polymetis_client = polysim.SimInterface(hz=hz)
    polymetis_client.register_arm_control(
        server_address="localhost:50051",
        state_callback=sim.get_state,
        action_callback=sim.apply_control,
        kp_joint=[25.0, 25.0],
        kd_joint=[5.0, 5.0],
        dof=1,
    )
    polymetis_client.register_step_callback(sim.step)

    polymetis_client.run()
