#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import hydra

from polymetis.robot_servers import GripperServerLauncher


@hydra.main(config_name="launch_gripper")
def main(cfg):
    if cfg.gripper:
        pid = os.fork()
    else:
        pid = os.getpid()  # doesn't fork so only the server gets launched

    if pid > 0:
        # Run server
        gripper_server = GripperServerLauncher(cfg.ip, cfg.port)
        gripper_server.run()

    else:
        # Run client (does not run if gripper=none)
        gripper_client = hydra.utils.instantiate(cfg.gripper)
        gripper_client.run()


if __name__ == "__main__":
    main()
