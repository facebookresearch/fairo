#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from polymetis import RobotInterface


def output_episode_stats(episode_name, robot_states):
    latency_arr = np.array(
        [robot_state.prev_controller_latency_ms for robot_state in robot_states]
    )
    latency_mean = np.mean(latency_arr)
    latency_std = np.std(latency_arr)
    latency_max = np.max(latency_arr)
    latency_min = np.min(latency_arr)

    success_arr = np.array(
        [robot_state.prev_command_successful for robot_state in robot_states]
    )
    success_rate = np.mean(success_arr)

    print(
        f"{episode_name}: {latency_mean:.4f}/ {latency_std:.4f} / {latency_max:.4f} / {latency_min:.4f} / {100 * success_rate:.2f}%"
    )


if __name__ == "__main__":
    robot = RobotInterface()

    print(
        "Control loop latency stats in milliseconds (avg / std / max / min / success_rate): "
    )

    # Test joint PD
    robot_states = robot.move_to_joint_positions(robot.get_joint_positions())
    output_episode_stats("Joint PD", robot_states)

    # Test cartesian PD
    robot_states = robot.move_to_ee_pose(robot.get_ee_pose()[0])
    output_episode_stats("Cartesian PD", robot_states)
