"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

from droidlet.dialog.map_to_dialogue_object import is_safe
from ..droidlet_nsp_model_wrapper import DroidletNSPModelWrapper
from droidlet.shared_data_structs import MockOpt

TTAD_BERT_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../agents/craftassist/datasets/annotated_data/")
TTAD_BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../../../agents/craftassist/models/semantic_parser/")

"""This class tests safety checks using a preset list of blacklisted words.
"""


class SafetyTest(unittest.TestCase):
    def setUp(self):
        opts = MockOpt()
        opts.nsp_data_dir = TTAD_BERT_DATA_DIR
        opts.nsp_models_dir = TTAD_BERT_MODEL_DIR
        opts.no_ground_truth = False
        self.chat_parser = DroidletNSPModelWrapper(opts)


    def test_unsafe_word(self):
        is_safe = is_safe(self.chat_parser.safety_words, "bad Clinton")
        self.assertFalse(is_safe)

    def test_safe_word(self):
        is_safe = is_safe(self.chat_parser.safety_words, "build a house")
        self.assertTrue(is_safe)


if __name__ == "__main__":
    unittest.main()
