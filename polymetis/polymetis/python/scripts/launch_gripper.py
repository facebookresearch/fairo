#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging

import hydra

from polymetis.robot_servers import GripperServerLauncher
from polymetis.utils.grpc_utils import check_server_exists
from polymetis.utils.data_dir import BUILD_DIR

log = logging.getLogger(__name__)


@hydra.main(config_name="launch_gripper")
def main(cfg):
    log.info(f"Adding {BUILD_DIR} to $PATH")
    os.environ["PATH"] = BUILD_DIR + os.pathsep + os.environ["PATH"]

    if cfg.gripper:
        pid = os.fork()
    else:
        pid = os.getpid()  # doesn't fork so only the server gets launched

    if pid > 0:
        # Run server
        gripper_server = GripperServerLauncher(cfg.ip, cfg.port)
        gripper_server.run()

    else:  # (this block does not run if gripper=none)
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
