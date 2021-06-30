# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch

from polymetis import RobotInterface
import torchcontrol as toco

num_dofs = 7
time_to_go = 0.1


def run_unending_policy(robot, policy, time_to_go):
    robot.send_torch_policy(policy, blocking=False)
    time.sleep(time_to_go)
    robot.terminate_current_policy()


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )
    robot.go_home()
    time.sleep(0.5)

    # Params
    hz = robot.metadata.hz
    robot_model = robot.robot_model
    joint_pos_current = robot.get_joint_angles()
    time_horizon = int(time_to_go * hz)

    # Run torchcontrol policies
    print("=== torchcontrol.policies.JointImpedanceControl ===")
    policy = toco.policies.JointImpedanceControl(
        joint_pos_current=joint_pos_current,
        Kp=torch.zeros(num_dofs, num_dofs),
        Kd=torch.zeros(num_dofs, num_dofs),
        robot_model=robot_model,
    )
    run_unending_policy(robot, policy, time_to_go)

    print("=== torchcontrol.policies.CartesianImpedanceControl ===")
    policy = toco.policies.CartesianImpedanceControl(
        joint_pos_current=joint_pos_current,
        Kp=torch.zeros(6, 6),
        Kd=torch.zeros(6, 6),
        robot_model=robot_model,
    )
    run_unending_policy(robot, policy, time_to_go)

    print("=== torchcontrol.policies.iLQR ===")
    policy = toco.policies.iLQR(
        Kxs=torch.zeros(time_horizon, num_dofs, 2 * num_dofs),
        x_desireds=torch.zeros(time_horizon, 2 * num_dofs),
        u_ffs=torch.zeros(time_horizon, num_dofs),
    )
    robot.send_torch_policy(policy)
