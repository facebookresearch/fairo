"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import re
import unittest
from copy import deepcopy
from droidlet.interpreter import process_spans_and_remove_fixed_value
from droidlet.perception.semantic_parsing.tests.test_y_print_parsing_report import (
    common_functional_commands,
    compare_full_dictionaries,
)
from .all_test_commands import INTERPRETER_POSSIBLE_ACTIONS, FILTERS, REFERENCE_OBJECTS

logical_form_before_processing = {
    "turn right": common_functional_commands["turn right"],
    "where are my keys": common_functional_commands["where are my keys"],
    "go forward": common_functional_commands["go forward"],
}

# FIXME! put these in the main file
logical_form_post_processing = {
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
            "where_clause": {"AND": [{"pred_text": "has_name", "obj_text": "keys"}]},
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


class TestInterpreterUtils(unittest.TestCase):
    def test_process_spans(self):
        for k, v in logical_form_before_processing.items():
            processed = deepcopy(v)
            original_words = re.split(r" +", k)
            lemmatized_words = original_words
            process_spans_and_remove_fixed_value(
                processed, original_words, lemmatized_words
            )  # process spans and fixed_values. Implemented in: interpreter_utils.
            assert compare_full_dictionaries(processed, logical_form_post_processing[k])

    def test_location_reference_object(self):
        def check_location_in_filters(action_dict):
            for key, value in action_dict.items():
                if key == "filters" and "location" in value:
                    return False
                elif type(value) == dict:
                    return check_location_in_filters(value)
            return True

        all_dicts = INTERPRETER_POSSIBLE_ACTIONS
        all_dicts.update(FILTERS)
        all_dicts.update(REFERENCE_OBJECTS)
        for key, action_dict in all_dicts.items():
            self.assertTrue(check_location_in_filters(action_dict))



if __name__ == "__main__":
    unittest.main()
