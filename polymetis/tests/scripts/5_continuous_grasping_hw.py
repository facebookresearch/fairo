# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from polymetis.utils.continuous_grasper import ManipulatorSystem


DEFAULT_MAX_ITERS = 3


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
    total_successes, total_tries = robot.continuously_grasp(max_iters)
    print(f"{total_successes}/{total_tries} successes")


if __name__ == "__main__":
    main(sys.argv)
