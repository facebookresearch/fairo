"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import signal
import subprocess
import time
import unittest
from ..craftassist_agent import CraftAssistAgent
from droidlet.shared_data_structs import MockOpt


class MyTestCase(unittest.TestCase):
    def test_agent_cuberite_connection(self):
        opts = MockOpt()
        opts.port = 25565
        repo_home = os.path.dirname(os.path.realpath(__file__))
        cuberite_process_path = os.path.join(repo_home, "../../../droidlet/lowlevel/minecraft/cuberite_process.py")
        with open("cuberite_log.txt", "w") as f:
            proc = subprocess.Popen(
                [f"python3 {cuberite_process_path} --config flat_world"],
                shell=True,
                stdout=f,
                preexec_fn=os.setsid,
            )
            time.sleep(60)  # let cuberite fully starts
            sa = CraftAssistAgent(opts)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(60)  # let cuberite fully terminates

        with open("cuberite_log.txt", "r") as f:
            cuberite_log = f.read()
            # check if agent has joined the game from cuberite log
            self.assertIn("has joined the game", cuberite_log)


if __name__ == "__main__":
    unittest.main()
