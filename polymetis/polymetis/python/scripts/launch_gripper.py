# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra

from polymetis.robot_drivers import robotiq_gripper


@hydra.main(config_name="launch_gripper")
def main(cfg):
    robotiq_gripper.run_server(cfg.ip, cfg.port, cfg.comport)


if __name__ == "__main__":
    main()
