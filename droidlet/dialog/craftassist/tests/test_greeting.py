"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import os
from agents.craftassist.tests.fake_agent import MockOpt
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase

GROUND_TRUTH_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../craftassist/agent/datasets/ground_truth/")

"""This class tests common greetings. Tests check whether the command executed successfully 
without world state changes; for correctness inspect chat dialogues in logging.
"""


class GreetingTest(BaseCraftassistTestCase):
    def setUp(self):
        opts = MockOpt()
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        opts.no_ground_truth = False
        super().setUp(agent_opts=opts)

    def test_hello(self):
        self.add_incoming_chat("hello", self.speaker)
        changes = self.flush()
        self.assertFalse(changes)

    def test_goodbye(self):
        self.add_incoming_chat("goodbye", self.speaker)
        changes = self.flush()
        self.assertFalse(changes)


if __name__ == "__main__":
    unittest.main()
