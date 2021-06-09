"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import os

# from droidlet.dialog.map_to_dialogue_object import get_greeting_reply
from droidlet.shared_data_structs import MockOpt
from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from droidlet.perception.semantic_parsing.droidlet_nsp_model_wrapper import DroidletNSPModelWrapper

GROUND_TRUTH_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../agents/craftassist/datasets/ground_truth/")

"""This class tests common greetings. Tests check whether the command executed successfully 
without world state changes; for correctness inspect chat dialogues in logging.
"""


class GreetingTest(unittest.TestCase):
    pass
    # def setUp(self):
    #     opts = MockOpt()
    #     opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
    #     self.dialogue_manager = DialogueManager(
    #         memory=self.memory,
    #         dialogue_object_classes={},
    #         dialogue_object_mapper=DialogueObjectMapper,
    #         opts=self.opts,
    #     )
    #
    # def test_hello(self):
    #     reply = get_greeting_reply(self.chat_parser.greetings, "hello")
    #     self.assertIn(reply, ["hi there!", "hello", "hey", "hi"])
    #
    # def test_goodbye(self):
    #     reply = get_greeting_reply(self.chat_parser.greetings, "goodbye")
    #     self.assertIn(reply, ["goodbye", "bye", "see you next time!"])


if __name__ == "__main__":
    unittest.main()
