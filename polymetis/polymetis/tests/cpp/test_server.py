#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

import hydra
from polymetis.utils.data_dir import PKG_ROOT_DIR, which

log = logging.getLogger(__name__)


@hydra.main(config_name="test_server")
def main(cfg):
    build_dir = os.path.abspath(os.path.join(PKG_ROOT_DIR, "..", "..", "build"))
    log.info(f"Adding {build_dir} to $PATH")
    os.environ["PATH"] = build_dir + os.pathsep + os.environ["PATH"]

    robot_client = hydra.utils.instantiate(cfg.robot_client)
    robot_client.run()


if __name__ == "__main__":
    main()
