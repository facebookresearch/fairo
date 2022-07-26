# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import time
import math
import numpy as np

from ipaddress import ip_address
from polymetis import RobotInterface
from polymetis import GripperInterface

positions = [0.0] * 10


def generate_sine_positions():
    # Generate sinusoidal positions based on current timestamp.
    for i in range(0, 6):
        ft = time.time() * 5 + i
        positions[i] = (0.5 * math.sin(ft) + 0.5) * 45 + 15
    positions[5] = positions[5] * -1
    return positions


def do_sine_wave(steps):
    # Move fingers through a sine wave. 'steps' is the number of joint
    # angle positions that will be iterated through.
    robot.go_home(time_to_go=1)
    for i in range(0, steps):
        desired_position = generate_sine_positions()
        robot.move_to_joint_positions(desired_position, 0.01)
        while True:
            actual_position = robot.get_joint_positions()
            diff = np.subtract(desired_position, actual_position)
            if all(abs(i) < 15 for i in diff):
                break


def do_rock_on():
    # Rock on pose
    robot.go_home(time_to_go=1)
    robot.move_to_joint_positions([0, 0, 0, 0, 90, -90, 0, 0, 0, 0], 1)
    time.sleep(0.1)
    robot.move_to_joint_positions([0, 90, 90, 0, 90, -90, 0, 0, 0, 0], 1)
    time.sleep(2)


def do_peace_sign():
    # Peace sign pose
    robot.go_home(time_to_go=1)
    robot.move_to_joint_positions([0, 0, 90, 90, 90, -90, 0, 0, 0, 0], 1)
    time.sleep(2)


def do_thumbs_up():
    # Thumbs up pose
    robot.go_home(time_to_go=1)
    robot.move_to_joint_positions([90, 90, 90, 90, 0, 0, 0, 0, 0, 0], 1)
    time.sleep(2)


def do_iterate_fingers():
    # Point pose
    positions = [
        [70, 90, 90, 90, 90, 0, 0, 0, 0, 0],
        [70, 90, 90, 90, 0, 0, 0, 0, 0, 0],
        [0, 90, 90, 90, 90, 0, 0, 0, 0, 0],
        [70, 50, 90, 90, 90, 0, 0, 0, 0, 0],
        [70, 90, 0, 90, 90, 0, 0, 0, 0, 0],
        [70, 90, 90, 0, 90, 0, 0, 0, 0, 0],
        [70, 90, 90, 90, 90, 0, 0, 0, 0, 0],
    ]
    robot.go_home(time_to_go=1)
    for angle_set in positions:
        robot.move_to_joint_positions(angle_set, 0.5)
        time.sleep(0.5)

    robot.go_home(time_to_go=1)


def do_grasps(n):
    # Do n iterations of a basic grasping motion.
    robot.go_home(time_to_go=1)
    for i in range(0, n):
        robot.move_to_joint_positions([90, 90, 90, 90, 90, -20, 0, 0, 0, 0], 0.2)
        time.sleep(1)
        robot.move_to_joint_positions([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.2)
        time.sleep(1)


if __name__ == "__main__":
    robot = RobotInterface(ip_address="127.0.0.1", port=50052)
    robot.get_robot_state()
    print(robot.get_joint_positions())
    print(robot.get_joint_velocities())
    robot.set_home_pose([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    while True:
        do_grasps(3)
        do_sine_wave(400)
        do_peace_sign()
        do_rock_on()
        do_thumbs_up()
        do_iterate_fingers()
