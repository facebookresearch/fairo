"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import sys

print(sys.path)
import unittest
from copy import deepcopy
from base_agent.dialogue_objects.dialogue_object_utils import process_spans
from test_y_print_parsing_report import common_functional_commands, compare_full_dictionaries

lf_pre = {
    "turn right": common_functional_commands["turn right"],
    "where are my keys": common_functional_commands["where are my keys"],
    "go forward": common_functional_commands["go forward"],
}
lf_post = {
    "turn right": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {"dance_type": {"body_turn": {"relative_yaw": "-90"}}, "action_type": "DANCE"}
        ],
    },
    "where are my keys": {
        "dialogue_type": "GET_MEMORY",
        "filters": {
            "output": {"attribute": "LOCATION"},
            "triples": [{"pred_text": "has_name", "obj_text": "keys"}],
        },
    },
    "go forward": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "location": {
                    "relative_direction": "FRONT",
                    "reference_object": {"special_reference": "AGENT"},
                },
                "action_type": "MOVE",
            }
        ],
    },
}


class TestProcessSpans(unittest.TestCase):
    def test_process_spans(self):
        for k, v in lf_pre.items():
            processed = deepcopy(v)
            process_spans(processed, k, k)
            assert compare_full_dictionaries(processed, lf_post[k])


if __name__ == "__main__":
    unittest.main()
