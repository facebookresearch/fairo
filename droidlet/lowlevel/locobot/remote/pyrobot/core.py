# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Robot:
    def __init__(
        self,
        robot_name,
        common_config={},
        parent=None,
    ):
        import pyrobot.cfg.habitat_config as habitat_config

        self.configs = habitat_config.get_cfg()
        self.configs.freeze()

        from pyrobot.habitat.simulator import HabitatSim
        from pyrobot.habitat.base import LoCoBotBase
        from pyrobot.habitat.camera import LoCoBotCamera

        self.simulator = HabitatSim(self.configs, **common_config)
        self.base = LoCoBotBase(self.configs, simulator=self.simulator, parent=parent)
        self.camera = LoCoBotCamera(self.configs, simulator=self.simulator)
