#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import numpy as np
import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from polymetis import RobotInterface
from polysim.grpc_sim_client import Spinner

META_CENTER_POSE = [0.55, 0.0, 0.45, 0.9383, 0.3442, 0.0, 0.0]
META_RADIUS = 0.15
UPDATE_HZ = 60
NUM_LOOPS = 1
TIME_TO_GO = 10
NUM_CHECKPOINTS = 20


def generate_trajectory(center_pose, radius, hz, num_loops, time_to_go):
    """Draw 3d Meta logo in free space"""
    num_steps = int(time_to_go * hz)
    ee_pose_traj = torch.empty([num_steps, 7])

    for i in range(num_steps):
        theta = i * num_loops * 2.0 * np.pi / num_steps

        # Position: Draw meta
        dx = 1.2 * radius * np.cos(theta)
        dy = 1.5 * radius * np.sin(theta)
        dz = radius * np.sin(theta * 2)
        ee_pose_traj[i, :3] = center_pose[:3] + torch.Tensor([dx, dy, dz])

        # Orientation: Only rotate wrist
        dr = R.from_rotvec(torch.Tensor([0.0, 0.0, np.pi * np.sin(theta) / 4.0]))
        ee_pose_traj[i, 3:] = (dr * R.from_quat(center_pose[3:])).as_quat()

    return ee_pose_traj


def run_offline_policy_from_traj(robot, traj):
    """Executes offline policy that executes trajectories in a similar fashion to robot.move_to_ee_pose"""
    num_steps = traj.shape[0]

    ee_pose_traj = []
    for i in range(num_steps):
        ee_pose = T.from_rot_xyz(
            rotation=R.from_quat(traj[i, 3:]),
            translation=traj[i, :3],
        )
        ee_pose_traj.append(ee_pose)

    ee_twist_traj = []
    for i in range(num_steps):
        pose0 = ee_pose_traj[max(i - 1, 0)]
        pose1 = ee_pose_traj[min(i + 1, num_steps - 1)]
        ee_twist = (pose1 * pose0.inv()).as_twist() * robot.metadata.hz
        ee_twist_traj.append(ee_twist)

    policy = toco.policies.EndEffectorTrajectoryExecutor(
        ee_pose_trajectory=ee_pose_traj,
        ee_twist_trajectory=ee_twist_traj,
        Kp=robot.Kx_default,
        Kd=robot.Kxd_default,
        robot_model=robot.robot_model,
        ignore_gravity=True,
    )

    return robot.send_torch_policy(policy)


def compare_traj(ref_traj, states_reached, robot_model, experiment_name=""):
    """Compute and print error between state log and the reference trajectory"""
    # Compute error
    pos_err = torch.zeros(len(states_reached))
    ori_err = torch.zeros(len(states_reached))
    for i in range(len(states_reached)):
        joint_pos_actual = torch.Tensor(states_reached[i][1].joint_positions)
        pose_actual = torch.cat(robot_model.forward_kinematics(joint_pos_actual))
        pose_desired = ref_traj[states_reached[i][0]]

        pos_diff = pose_desired[:3] - pose_actual[:3]
        ori_diff = R.from_quat(pose_actual[3:]).inv() * R.from_quat(pose_desired[3:])
        pos_err[i] = torch.linalg.norm(pos_diff)
        ori_err[i] = torch.linalg.norm(ori_diff.as_rotvec())

    # Compute & print stats
    pos_err_mean = torch.mean(pos_err)
    pos_err_std = torch.std(pos_err)
    ori_err_mean = torch.mean(ori_err)
    ori_err_std = torch.std(ori_err)

    print(f"=== {experiment_name} tracking results ===")
    print(f"\tPos error: mean={pos_err_mean:.4f}, std={pos_err_std:.4f}")
    print(f"\tOri error: mean={ori_err_mean:.4f}, std={ori_err_std:.4f}")


if __name__ == "__main__":
    robot = RobotInterface()
    robot.go_home()

    # Generate trajectory
    pose_traj = generate_trajectory(
        torch.Tensor(META_CENTER_POSE),
        META_RADIUS,
        UPDATE_HZ,
        NUM_LOOPS,
        TIME_TO_GO,
    )
    num_steps = pose_traj.shape[0]

    # Offline: Send entire trajectory as controller
    robot.move_to_ee_pose(pose_traj[0, :3], pose_traj[0, 3:])

    states_reached = []
    for i in range(NUM_CHECKPOINTS):
        waypoint_idx = min(int(((i + 1) / NUM_CHECKPOINTS) * num_steps), num_steps - 1)
        pose_desired = pose_traj[waypoint_idx, :]
        state_log = robot.move_to_ee_pose(pose_desired[:3], pose_desired[3:])
        states_reached.append((waypoint_idx, state_log[-1]))
    compare_traj(
        pose_traj, states_reached, robot.robot_model, "Offline (move_to_ee_pose)"
    )

    # Online: Send pose updates to impedance controller
    robot.move_to_ee_pose(pose_traj[0, :3], pose_traj[0, 3:])

    robot.start_cartesian_impedance()
    states_reached = []
    i_curr = 0
    for i in range(NUM_CHECKPOINTS):
        i_target = min(int(((i + 1) / NUM_CHECKPOINTS) * num_steps), num_steps - 1)
        spinner = Spinner(UPDATE_HZ)
        for pose in pose_traj[i_curr:i_target, :]:
            robot.update_desired_ee_pose(pose[:3], pose[3:])
            spinner.spin()
        time.sleep(0.5)  # wait for steady state

        states_reached.append((i_target - 1, robot.get_robot_state()))
        i_curr = i_target

    robot.terminate_current_policy()
    compare_traj(
        pose_traj, states_reached, robot.robot_model, "Online (update_desired_ee_pose)"
    )
