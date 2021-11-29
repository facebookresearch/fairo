"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import os
from droidlet.shared_data_structs import MockOpt
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper

GROUND_TRUTH_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../droidlet/artifacts/datasets/ground_truth/")

"""This class tests common greetings. Tests check whether the command executed successfully 
without world state changes; for correctness inspect chat dialogues in logging.
"""


class GreetingTest(unittest.TestCase):
    def setUp(self):
        opts = MockOpt()
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        self.dialogue_object_mapper = DialogueObjectMapper(
            dialogue_object_classes={},
            opts=opts,
            low_level_interpreter_data={},
            dialogue_manager=None
        )

    def test_hello(self):
        reply = self.dialogue_object_mapper.get_greeting_reply("hello")
        self.assertIn(reply, ["hi there!", "hello", "hey", "hi"])

    def test_goodbye(self):
        reply = self.dialogue_object_mapper.get_greeting_reply("goodbye")
        self.assertIn(reply, ["goodbye", "bye", "see you next time!"])


if __name__ == "__main__":
    unittest.main()
