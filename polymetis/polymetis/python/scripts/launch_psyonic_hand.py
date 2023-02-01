#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra


@hydra.main(config_name="launch_psyonic_hand")
def main(cfg):
    gripper_server = hydra.utils.instantiate(cfg.gripper)
    gripper_server.run()


if __name__ == "__main__":
    main()
