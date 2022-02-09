#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging
import polymetis
import a0

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


if __name__ == "__main__":
    publisher = a0.Publisher("latest_robot_state")

    while True:
        log.warning("Connecting to server...")
        try:
            robot_interface = polymetis.RobotInterface()
        except Exception as e:
            log.error(f"Failed to connect to server: {e}")
            time.sleep(1)
            continue
        log.info("Connected.")

        try:
            while True:
                log.info("Attempting to retrieve latest robot state...")
                curr_state = robot_interface.get_robot_state()
                publisher.pub(f"{curr_state.timestamp.seconds}")
        except Exception as e:
            log.error(f"Failed to retrieve robot state! Attempting to reconnect...")
