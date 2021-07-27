"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import unittest
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from droidlet.shared_data_structs import MockOpt

TTAD_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../agents/craftassist/models/semantic_parser/"
)
TTAD_BERT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../agents/craftassist/datasets/annotated_data/"
)
GROUND_TRUTH_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../agents/craftassist/datasets/ground_truth/"
)


class TestValidateParseTree(unittest.TestCase):
    def setUp(self):
        opts = MockOpt()
        opts.nsp_data_dir = TTAD_BERT_DATA_DIR
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        opts.nsp_models_dir = TTAD_MODEL_DIR
        opts.no_ground_truth = False
        self.chat_parser = NSPQuerier(opts)

    def test_validate_bad_json(self):
        # Don't print debug info on failure
        is_valid_json = self.chat_parser.validate_parse_tree(parse_tree={}, debug=False)
        self.assertFalse(is_valid_json)

    def test_validate_array_span_json(self):
        action_dict = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action_sequence": [
                {
                    "action_type": "BUILD",
                    "schematic": {
                        "text_span": [0, [5, 5]],
                        "filters": {
                            "where_clause": {
                                "AND": [{"pred_text": "has_name", "obj_text": [0, [5, 5]]}]
                            }
                        },
                    },
                }
            ],
        }
        is_valid_json = self.chat_parser.validate_parse_tree(action_dict)
        self.assertTrue(is_valid_json)

    def test_validate_string_span_json(self):
        action_dict = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action_sequence": [
                {
                    "action_type": "DANCE",
                    "dance_type": {
                        "look_turn": {
                            "location": {
                                "reference_object": {
                                    "filters": {
                                        "where_clause": {
                                            "AND": [{"pred_text": "has_name", "obj_text": "cube"}]
                                        }
                                    }
                                }
                            }
                        }
                    },
                }
            ],
        }
        is_valid_json = self.chat_parser.validate_parse_tree(action_dict)
        self.assertTrue(is_valid_json)


if __name__ == "__main__":
    unittest.main()
