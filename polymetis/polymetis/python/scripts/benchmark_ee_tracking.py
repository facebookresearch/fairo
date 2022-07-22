#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from torchcontrol import Rotation as R
from polymetis import RobotInterface
from polysim.grpc_sim_client import Spinner

META_CENTER_POSE = [0.55, 0.0, 0.5, 0.9383, 0.3442, 0.0, 0.0]
META_RADIUS = 0.3
UPDATE_HZ = 60


def generate_trajectory(center_pose, radius, hz, time_to_go):
    """Draw 3d Meta logo in free space"""
    num_steps = int(time_to_go * hz)
    ee_pose_traj = torch.empty([num_steps, 7])

    for i in range(num_steps):
        theta = 4 * np.pi / num_steps

        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)
        dz = radius * np.sin(theta * 2)
        ee_pose_traj[i, :3] = center_pose[:3] + torch.Tensor([dx, dy, dz])

        dr_wrist = R.from_rotvec(torch.Tensor([0.0, 0.0, np.sin(theta / 2.0)]))
        dr_sway = R.from_rotvec([np.sin(theta), 0.0, 0.0])
        dr = dr_sway * dr_wrist
        ee_pose_traj[i, 3:] = (dr * R.from_quat(center_pose[3:])).as_quat()

    return ee_pose_traj


def compare_traj(traj, ref_traj, experiment_name=""):
    assert traj.shape[0] == ref_traj.shape[0]

    pos_err = torch.zeros(traj.shape[0])
    ori_err = torch.zeros(traj.shape[0])
    for i in range(shape[0]):
        pos_diff = traj[i, :3] - ref_traj[i, :3]
        ori_diff = R.from_quat(traj[i, 3:]).inv() * R.from_quat(traj[i, 3:])
        pos_err[i] = torch.linalg.norm(pos_err)
        ori_err[i] = torch.linalg.norm(ori_err.as_rotvec())

    # Print stats TODO


if __name__ == "__main__":
    robot = RobotInterface()

    # Offline tracking: Send entire trajectory as controller
    pose_traj_offline = generate_trajectory(
        META_CENTER_POSE, META_RADIUS, robot.metadata.hz, 12
    )
    robot.move_to_ee_pose(pose_traj[0, :3], pose_traj[0, 3:])

    state_log = None  # TODO

    compare_traj(state_log, pose_traj_offline, "offline")

    # Online tracking: Send trajectory updates to impedance controller
    pose_traj_online = generate_trajectory(META_CENTER_POSE, META_RADIUS, UPDATE_HZ, 12)
    robot.move_to_ee_pose(pose_traj[0, :3], pose_traj[0, 3:])

    robot.start_cartesian_impedance()
    spinner = Spinner(UPDATE_HZ)
    for pose in pose_traj:
        robot.update_desired_ee_pose(pose[:3], pose[3:])
        spinner.spin()

    state_log = robot.terminate_current_policy()

    compare_traj(state_log, pose_traj_offline, "online")
