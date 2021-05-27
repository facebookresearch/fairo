import unittest
import subprocess
import time
from ..craftassist_agent import CraftAssistAgent
from .fake_agent import MockOpt

class MyTestCase(unittest.TestCase):
    def test_agent_with_cuberite(self):
        opts = MockOpt()
        opts.port = 25565
        cuberite_process = "droidlet/lowlevel/minecraft/cuberite_process.py "
        p = subprocess.Popen([f"python {cuberite_process} --config flat_world"], shell=True)
        time.sleep(1) # let cuberite fully start
        sa = CraftAssistAgent(opts)


if __name__ == '__main__':
    unittest.main()
