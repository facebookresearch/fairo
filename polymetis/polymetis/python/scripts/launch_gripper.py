#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import hydra

from polymetis.robot_servers import GripperServerLauncher
from polymetis.utils.grpc_utils import check_server_exists


@hydra.main(config_name="launch_gripper")
def main(cfg):
    if os.fork() > 0:
        # Run server
        gripper_server = GripperServerLauncher(cfg.ip, cfg.port)
        gripper_server.run()

    else:
        # Wait for server to launch
        t0 = time.time()
        while not check_server_exists(cfg.ip, cfg.port):
            time.sleep(0.1)
            if time.time() - t0 > cfg.timeout:
                raise ConnectionError("Robot client: Unable to locate server.")

        # Run client
        gripper_client = hydra.utils.instantiate(cfg.gripper)
        gripper_client.run()


if __name__ == "__main__":
    main()
