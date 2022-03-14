# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

from polymetis.utils.continuous_grasper import ManipulatorSystem

DEFAULT_MAX_ITERS = 3


def main(max_iterations, **kwargs):
    try:
        max_iters = int(max_iterations)
    except ValueError as exc:
        print(f"Malformed iteration count, using defaults {DEFAULT_MAX_ITERS}")
        max_iters = DEFAULT_MAX_ITERS
    robot = ManipulatorSystem(robot_kwargs=kwargs, gripper_kwargs=kwargs)
    total_successes, total_tries = robot.continuously_grasp(max_iters)
    print(f"{total_successes}/{total_tries} successes")


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print(
            "Usage: python 4_continuous_grasping.py <max_iterations> <robot and / or gripper parameters>"
        )
    else:
        main(**dict(arg.split("=") for arg in sys.argv[1:]))
