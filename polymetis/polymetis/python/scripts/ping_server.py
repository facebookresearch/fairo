#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging
import polymetis

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


if __name__ == "__main__":
    log.warning("Connecting to server...")
    robot_interface = polymetis.RobotInterface()
    log.info("Connected.")

    log.info("Attempting to retrieve latest robot state...")
    curr_state = robot_interface.get_robot_state()
    time_diff = time.time() - curr_state.timestamp.seconds
    log.info(f"Robot state retrieved within {time_diff}s.")

    log.info("Timestamp: ")
    log.info(f"{curr_state.timestamp}")

    num_seconds_stale = 1
    assert time_diff < num_seconds_stale, f"Robot state too stale by {time_diff}, expected within {num_seconds_stale}."
