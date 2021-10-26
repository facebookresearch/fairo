"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
from droidlet.shared_data_structs import MockOpt
from agents.craftassist.craftassist_agent import CraftAssistAgent


class Opt:
    pass


class BaseAgentTest(unittest.TestCase):
    def test_init_agent(self):
        opts = MockOpt()
        CraftAssistAgent(opts)


if __name__ == "__main__":
    unittest.main()
