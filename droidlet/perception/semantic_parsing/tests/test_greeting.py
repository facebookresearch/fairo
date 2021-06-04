"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import os
from droidlet.shared_data_structs import MockOpt
from ..droidlet_nsp_model_wrapper import DroidletNSPModelWrapper

GROUND_TRUTH_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../agents/craftassist/datasets/ground_truth/")

"""This class tests common greetings. Tests check whether the command executed successfully 
without world state changes; for correctness inspect chat dialogues in logging.
"""


class GreetingTest(unittest.TestCase):
    def setUp(self):
        opts = MockOpt()
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        self.chat_parser = DroidletNSPModelWrapper(opts)

    def test_hello(self):
        self.chat_parser.is_greeting("hello")
        pass

    def test_goodbye(self):
        self.chat_parser.is_greeting("goodbye")
        pass


if __name__ == "__main__":
    unittest.main()
