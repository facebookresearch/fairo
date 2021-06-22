"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import os
from droidlet.shared_data_structs import MockOpt
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper

GROUND_TRUTH_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../agents/craftassist/datasets/ground_truth/")

"""This class tests safety checks using a preset list of blacklisted words.
"""


class SafetyTest(unittest.TestCase):
    def setUp(self):
        opts = MockOpt()
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        self.dialogue_object_mapper = DialogueObjectMapper(
            dialogue_object_classes={},
            opts=opts,
            dialogue_manager=None
        )

    def test_unsafe_word(self):
        is_safe = self.dialogue_object_mapper.is_safe("bad Clinton")
        self.assertFalse(is_safe)

    def test_safe_word(self):
        is_safe = self.dialogue_object_mapper.is_safe("build a house")
        self.assertTrue(is_safe)


if __name__ == "__main__":
    unittest.main()
