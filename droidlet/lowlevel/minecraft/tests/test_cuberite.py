"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from droidlet.lowlevel.minecraft.cuberite_process import CuberiteProcess, create_workdir, repo_home
from os import path
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.ping_cuberite import ping


class CuberiteBasicTest(unittest.TestCase):

    def test_create_workdir(self):
        # diverse_world
        plugins = ["debug", "chatlog", "point_blocks"]
        workdir = create_workdir(config_name="flat_world",
                                 seed=0, game_mode="creative",
                                 port=25565,
                                 plugins=plugins,
                                 place_blocks_yzx=None,
                                 workdir_root=None)
        self.assertIsNotNone(workdir)
        self.assertIn("tmp", workdir)
        self.assertIn("cuberite", workdir)

    def test_check_cuberite_executable(self):
        popen = [repo_home + "/server/cuberite/Server/Cuberite"]
        self.assertTrue(path.exists(popen[0]))

    def test_launch_cuberite_process(self):
        plugins = ["debug", "chatlog", "point_blocks"]
        p = CuberiteProcess(
            config_name="flat_world",
            seed=0,
            game_mode="creative",
            port=25565,
            plugins=plugins)
        try:
            ping("localhost", 25565)
            print("Successfully pinged an existing cuberite instance.")
        except:
            self.fail("ping cuberite raised exception unexpectedly!")
        p.destroy()
        # Assert that ping raises an exception.
        self.assertRaises(ConnectionRefusedError, ping, **{"host": "localhost", "port": 25565})

if __name__ == '__main__':
    unittest.main()
