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
PREGRASP_HEIGHT = 0.55
GRASP_HEIGHT = 0.25
PLANNER_DT = 0.02


class ManipulatorSystem:
    def __init__(self):
        self.arm = RobotInterface()
        self.gripper = GripperInterface()
        time.sleep(0.5)

        # Set continuous control policy
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
        # Go home
        self.arm.go_home()

        # Send PD controller
        joint_pos_current = self.arm.get_joint_angles()
        policy = toco.policies.CartesianImpedanceControl(
            joint_pos_current=joint_pos_current,
            Kp=torch.Tensor(self.arm.metadata.Kx_default),
            Kd=torch.Tensor(self.arm.metadata.Kxd_default),
            robot_model=self.arm.robot_model,
        )
        self.arm.send_torch_policy(policy, blocking=False)

    def move_to(self, pos, quat, time_to_go=2.0):
        # Plan trajectory
        pos_curr, quat_curr = self.arm.pose_ee()
        N = int(time_to_go / PLANNER_DT)
        waypoints = toco.planning.generate_cartesian_space_min_jerk(
            start=T.from_rot_xyz(R.from_quat(quat_curr), pos_curr),
            goal=T.from_rot_xyz(R.from_quat(quat), pos),
            time_to_go=time_to_go,
            hz=self.arm.metadata.hz,
        )

        # Execute trajectory
        t0 = time.time()
        t_target = t0
        for i in range(N):
            # Update traj
            ee_pos_desired = waypoints[i]["pose"].translation()
            ee_quat_desired = waypoints[i]["pose"].rotation().as_quat()
            # ee_twist_desired = waypoints[i]["twist"]
            self.arm.update_current_policy(
                {
                    "ee_pos_desired": ee_pos_desired,
                    "ee_quat_desired": ee_quat_desired,
                    # "ee_vel_desired": ee_twist_desired[:3],
                    # "ee_rvel_desired": ee_twist_desired[3:],
                }
            )

            # Check if policy terminated due to issues
            if self.arm.get_previous_interval().end != -1:
                print("Interrupt detected. Reinstantiating control policy...")
                time.sleep(3)
                self.reset_policy()
                break

            # Spin once
            t_target += PLANNER_DT
            t_remaining = t_target - time.time()
            time.sleep(max(t_remaining, 0.0))

        # Wait for robot to stabilize
        time.sleep(0.2)

    def close_gripper(self):
        self.gripper.grasp(speed=0.1, force=1.0)
        time.sleep(0.5)

        # Check state
        state = self.gripper.get_state()
        assert state.width < state.max_width

    def open_gripper(self):
        max_width = self.gripper.get_state().max_width
        self.gripper.goto(width=max_width, speed=0.1, force=1.0)
        time.sleep(0.5)

        # Check state
        state = self.gripper.get_state()
        assert np.allclose(state.width, state.max_width, atol=1e-3)

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

        # Lower (slower than other motions to prevent sudden collisions)
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, GRASP_HEIGHT)
        self.move_to(pos, quat, time_to_go=4.0)

        # Grasp
        self.close_gripper()

        # Lift to pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, PREGRASP_HEIGHT)
        self.move_to(pos, quat)

        # Move to new pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose1, PREGRASP_HEIGHT)
        self.move_to(pos, quat)

        # Release
        self.open_gripper()

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
