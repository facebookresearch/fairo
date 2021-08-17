# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys

import numpy as np
import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from polymetis import RobotInterface, GripperInterface


DEFAULT_MAX_ITERS = 3

# Sampling params
GP_RANGE_UPPER = [0.7, 0.1, np.pi / 2]
GP_RANGE_LOWER = [0.4, -0.1, -np.pi / 2]

# Grasp params
REST_POSE = ([0.5, 0.0, 0.7], [1.0, 0.0, 0.0, 0.0])
PREGRASP_HEIGHT = 0.4
GRASP_HEIGHT = 0.25
PLANNER_DT = 0.02


class ManipulatorSystem:
    def __init__(self):
        self.arm = RobotInterface()
        self.gripper = GripperInterface()
        time.sleep(0.5)

        # Send PD controller
        self.reset_policy()

        # Reset to rest pose
        self.rest_pos = torch.Tensor(REST_POSE[0])
        self.rest_quat = torch.Tensor(REST_POSE[1])
        self.reset()

    def __del__(self):
        self.arm.terminate_current_policy()

    def reset(self, time_to_go=2.0):
        self.move_to(self.rest_pos, self.rest_quat, time_to_go)
        self.open_gripper()

    def reset_policy(self):
        joint_pos_current = self.arm.get_joint_angles()
        policy = toco.policies.JointImpedanceControl(
            joint_pos_current=joint_pos_current,
            Kp=self.arm.metadata.default_Kq,
            Kd=self.arm.metadata.default_Kqd,
            robot_model=self.arm.robot_model,
        )
        self.arm.send_torch_policy(policy, blocking=False)

    def move_to(self, pos, quat, time_to_go=3.0):
        # Plan trajectory
        joint_pos_current = self.arm.get_joint_angles()
        N = int(time_to_go / PLANNER_DT)
        plan = toco.modules.CartesianSpaceMinJerkJointPlanner(
            joint_pos_start=joint_pos_current,
            ee_pose_goal=T.from_rot_xyz(R.from_quat(quat), pos),
            steps=N,
            time_to_go=time_to_go,
            robot_model=self.arm.robot_model,
        )

        # Execute trajectory
        t0 = time.time()
        t_target = t0
        for i in range(N):
            # Update traj
            joint_pos_desired, _, _ = plan(i)
            self.arm.update_current_policy({"joint_pos_desired": joint_pos_desired})

            # Spin once
            t_target += PLANNER_DT
            t_remaining = t_target - time.time()
            time.sleep(max(t_remaining, 0.0))

        # Wait for robot to stabilize
        time.sleep(0.2)

    def close_gripper(self):
        self.gripper.goto(pos=0, vel=0.1, force=1.0)
        time.sleep(0.5)

    def open_gripper(self):
        self.gripper.goto(pos=0.14, vel=0.1, force=1.0)
        time.sleep(0.5)

    def grasp_pose_to_pos_quat(self, grasp_pose, z):
        x, y, rz = grasp_pose
        pos = torch.Tensor([x, y, z])
        quat = (
            R.from_rotvec(torch.Tensor([0, 0, rz])) * R.from_quat(self.rest_quat)
        ).as_quat()

        return pos, quat

    def grasp(self, grasp_pose0, grasp_pose1):
        # Move to pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, PREGRASP_HEIGHT)
        self.move_to(pos, quat)

        # Lower
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, GRASP_HEIGHT)
        self.move_to(pos, quat)

        # Grasp
        self.close_gripper()

        # Lift to pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, PREGRASP_HEIGHT)
        self.move_to(pos, quat, time_to_go=2.0)

        # Move to new pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose1, PREGRASP_HEIGHT)
        self.move_to(pos, quat)

        # Release
        self.open_gripper()

        # Check if policy terminated due to issues
        if self.arm.get_previous_interval().end != -1:
            print("Interrupt detected. Reinstantiating control policy...")
            time.sleep(3)
            self.reset_policy()

        # Reset
        self.reset()


def uniform_sample(lower, upper):
    return lower + (upper - lower) * torch.rand_like(lower)


def main(argv):
    if len(argv) > 1:
        try:
            max_iters = int(argv[1])
        except ValueError as exc:
            print("Usage: python 5_continuous_grasping.py <max_iterations>")
            return
    else:
        max_iters = DEFAULT_MAX_ITERS

    # Initialize interfaces
    robot = ManipulatorSystem()

    # Setup sampling
    gp_range_upper = torch.Tensor(GP_RANGE_UPPER)
    gp_range_lower = torch.Tensor(GP_RANGE_LOWER)

    # Perform grasping
    i = 0
    try:
        while True:
            # Sample grasp
            grasp_pose0 = uniform_sample(gp_range_lower, gp_range_upper)
            grasp_pose1 = uniform_sample(gp_range_lower, gp_range_upper)

            # Perform grasp
            print(f"Grasp {i + 1}: grasp={grasp_pose0}, release={grasp_pose1}")
            robot.grasp(grasp_pose0, grasp_pose1)

            # Loop termination
            i += 1
            if max_iters > 0 and i >= max_iters:
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main(sys.argv)
