# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from polymetis import RobotInterface


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    robot.go_home()

    # Get ee pose
    ee_pos, ee_quat = robot.pose_ee()
    print(f"Current ee position: {ee_pos}")
    print(f"Current ee orientation: {ee_quat}  (xyzw)")

    # Command robot to ee pose (move ee downwards)
    # note: can also be done with robot.move_ee_xyz
    delta_ee_pos_desired = torch.Tensor([0.0, 0.0, -0.1])
    ee_pos_desired = ee_pos + delta_ee_pos_desired
    print(f"\nMoving ee pos to: {ee_pos_desired} ...\n")
    state_log = robot.set_ee_pose(
        position=ee_pos_desired, orientation=None, time_to_go=2.0
    )

    # Get updated ee pose
    ee_pos, ee_quat = robot.pose_ee()
    print(f"New ee position: {ee_pos}")
    print(f"New ee orientation: {ee_quat}  (xyzw)")
