#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import hydra

from polymetis.robot_servers import GripperServerLauncher


@hydra.main(config_name="launch_gripper")
def main(cfg):
    if os.fork() > 0:
        # Run server
        gripper_server = GripperServerLauncher(cfg.ip, cfg.port)
        gripper_server.run()

    else:
        # Run client
        print(cfg.gripper)
        gripper_client = hydra.utils.instantiate(cfg.gripper)
        gripper_client.run()


if __name__ == "__main__":
    main()
