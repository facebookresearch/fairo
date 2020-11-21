"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

from base_craftassist_test_case import BaseCraftassistTestCase
from fake_agent import MockOpt

GROUND_TRUTH_DATA_DIR = os.path.join(os.path.dirname(__file__), "../agent/datasets/ground_truth/")

"""This class tests safety checks using a preset list of blacklisted words.
"""


class SafetyTest(BaseCraftassistTestCase):
    def setUp(self):
        opts = MockOpt()
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        opts.no_ground_truth = False
        super().setUp(agent_opts=opts)

    def test_unsafe_word(self):
        is_safe = self.agent.dialogue_manager.is_safe("bad Clinton")
        self.assertFalse(is_safe)

    def test_safe_word(self):
        is_safe = self.agent.dialogue_manager.is_safe("build a house")
        self.assertTrue(is_safe)

    def test_dialogue_manager(self):
        self.add_incoming_chat("bad Clinton", self.speaker)
        changes = self.flush()
        self.assertFalse(changes)


if __name__ == "__main__":
    unittest.main()
