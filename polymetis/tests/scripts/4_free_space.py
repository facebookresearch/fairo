# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys

import torch

from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T

POS_RANGE_UPPER = [0.6, 0.2, 0.6]
POS_RANGE_LOWER = [0.4, -0.2, 0.2]
ORI_RANGE = [0.5, 0.5, 0.5]

DEFAULT_MAX_ITERS = 3


def uniform_sample(lower, upper):
    return lower + (upper - lower) * torch.rand_like(lower)


def main(argv):
    if len(argv) > 1:
        try:
            max_iters = int(argv[1])
        except ValueError as exc:
            print("Usage: python 4_free_space.py <max_iterations>")
            return
    else:
        max_iters = DEFAULT_MAX_ITERS

    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )
    hz = robot.metadata.hz
    time.sleep(0.5)

    # Get reference state
    pos0, quat0 = robot.pose_ee()

    # Setup sampling
    pos_range_upper = torch.Tensor(POS_RANGE_UPPER)
    pos_range_lower = torch.Tensor(POS_RANGE_LOWER)
    ori_range = torch.Tensor(ORI_RANGE)

    # Random movements
    i = 0
    try:
        while True:
            robot.go_home()

            # Sample pose
            pos_sampled = uniform_sample(pos_range_lower, pos_range_upper)
            ori_sampled = R.from_quat(quat0) * R.from_rotvec(
                uniform_sample(-ori_range, ori_range)
            )

            # Move to pose
            print(
                f"Moving to random pose ({i + 1}): pos={pos_sampled}, quat={ori_sampled.as_quat()}"
            )
            state_log = robot.set_ee_pose(
                position=pos_sampled,
                orientation=ori_sampled.as_quat(),
            )
            print(f"\t Episode length: {len(state_log)}")

            # Loop termination
            i += 1
            if max_iters > 0 and i >= max_iters:
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main(sys.argv)
