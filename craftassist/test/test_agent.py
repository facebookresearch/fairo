"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
from .fake_agent import MockOpt

from craftassist.agent.craftassist_agent import CraftAssistAgent


class Opt:
    pass


class BaseAgentTest(unittest.TestCase):
    def test_init_agent(self):
        opts = MockOpt()
        CraftAssistAgent(opts)


if __name__ == "__main__":
    unittest.main()
