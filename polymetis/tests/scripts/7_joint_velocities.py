# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch

from polymetis import RobotInterface


def test_new_joint_vel(state_log, joint_vel_desired):
    for i, state in enumerate(state_log):
        assert torch.allclose(
            torch.Tensor(state.joint_velocities),
            joint_vel_desired,
            rtol=0.01,
            atol=0.01,
        ), f"""iteration {i}:
                        measured velocities {torch.Tensor(state.joint_velocities)},
                        desired velocities {joint_vel_desired}"""


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )
    robot.go_home()
    time.sleep(0.5)

    # Joint velocity control
    print("=== RobotInterface.start_joint_velocity_control ===")

    joint_vel_desired = torch.Tensor([0.01, -0.01, -0.01, 0.0, 0.0, 0.01, 0.01])

    robot.start_joint_velocity_control(joint_vel_desired)
    time.sleep(0.6)
    state_log = robot.terminate_current_policy()

    hz = robot.metadata.hz
    test_new_joint_vel(state_log[int(hz / 2) :], joint_vel_desired)

    new_joint_vel_desired = torch.Tensor([-0.01, 0.01, 0.0, -0.01, 0.01, -0.01, -0.01])
    robot.start_joint_velocity_control(joint_vel_desired)
    time.sleep(0.1)
    robot.update_desired_joint_velocities(new_joint_vel_desired)
    time.sleep(1)
    state_log = robot.terminate_current_policy()
    test_new_joint_vel(state_log[int(1 * hz) :], new_joint_vel_desired)
