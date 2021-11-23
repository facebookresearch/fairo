# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch

from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
from utils import check_episode_log


def test_new_ee_pose(robot, ee_pos_desired, ee_quat_desired):
    ee_pos, ee_quat = robot.pose_ee()
    print(f"Desired ee pose: pos={ee_pos_desired}, quat={ee_quat_desired}")
    print(f"New ee pose: pos={ee_pos}, quat={ee_quat}")

    assert torch.allclose(ee_pos, ee_pos_desired, atol=0.005)
    assert (
        R.from_quat(ee_quat).inv() * R.from_quat(ee_quat_desired)
    ).magnitude() < 0.0174  # 1 degree

    return ee_pos, ee_quat


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )
    hz = robot.metadata.hz
    robot.go_home()
    time.sleep(0.5)

    # Get joint positions
    joint_pos = robot.get_joint_angles()
    ee_pos, ee_quat = robot.pose_ee()
    print(f"Initial ee pose: pos={ee_pos}, quat={ee_quat}")

    # Go to ee_pose
    print("=== RobotInterface.set_ee_pose ===")
    ee_pos_desired = ee_pos + torch.Tensor([0.0, 0.05, -0.05])
    ee_quat_desired = torch.Tensor([1, 0, 0, 0])  # pointing straight down
    time_to_go = 4.0

    state_log = robot.set_ee_pose(
        ee_pos_desired, ee_quat_desired, time_to_go=time_to_go
    )
    time.sleep(0.5)

    ee_pos, ee_quat = test_new_ee_pose(robot, ee_pos_desired, ee_quat_desired)
    check_episode_log(state_log, int(time_to_go * hz))

    # Go to ee_pose (larger movement)
    print("=== RobotInterface.set_ee_pose ===")
    ee_pos_desired = ee_pos + torch.Tensor([0.1, -0.1, 0.2])
    ee_quat_desired = torch.Tensor(
        [0.7071, 0, 0, 0.7071]
    )  # rotate by 90 degrees around x axis
    time_to_go = 6.0

    state_log = robot.set_ee_pose(
        ee_pos_desired, ee_quat_desired, time_to_go=time_to_go
    )
    time.sleep(0.5)

    ee_pos, ee_quat = test_new_ee_pose(robot, ee_pos_desired, ee_quat_desired)
    check_episode_log(state_log, int(time_to_go * hz))

    # Move by delta ee pose
    print("=== RobotInterface.move_ee_xyz ===")
    delta_ee_pos_desired = torch.Tensor([-0.1, 0.0, -0.1])
    ee_pos_desired = ee_pos + delta_ee_pos_desired
    ee_quat_desired = ee_quat
    time_to_go = 6.0

    state_log = robot.move_ee_xyz(
        delta_ee_pos_desired, time_to_go=time_to_go, use_orient=False
    )
    time.sleep(0.5)

    ee_pos, ee_quat = test_new_ee_pose(robot, ee_pos_desired, ee_quat_desired)
    check_episode_log(state_log, int(time_to_go * hz))
