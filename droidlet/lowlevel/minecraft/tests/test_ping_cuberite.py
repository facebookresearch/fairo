"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import socket
import unittest
import subprocess
from droidlet.lowlevel.minecraft.cuberite_process import CuberiteProcess, create_workdir

class CuberiteBasicTest(unittest.TestCase):
    def test_create_workdir(self):
        # diverse_world
        plugins = ["debug", "chatlog", "point_blocks"]
        create_workdir(config_name="flat_world", seed=0, game_mode="creative", port=25565, plugins=plugins, place_blocks_yzx=None, workdir_root=None)
    # CuberiteProcess() -> launch
    # ping_cuberite
    # destroy ?

if __name__ == '__main__':
    unittest.main()
