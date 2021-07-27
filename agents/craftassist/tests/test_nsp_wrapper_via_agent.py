"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import unittest
from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.memory.dialogue_stack import DialogueStack
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from agents.loco_mc_agent import LocoMCAgent
from droidlet.shared_data_structs import MockOpt


# FIXME agent this test needs to move to the interpreter folder after
# dialogue_manager is properly split between agent and intepreter


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class FakeMemory:
    pass


class FakeAgent(LocoMCAgent):
    def __init__(self, opts):
        super(FakeAgent, self).__init__(opts)
        self.opts = opts

    def init_memory(self):
        m = FakeMemory()
        stack = DialogueStack()
        m.dialogue_stack = stack
        self.memory = m

    def init_physical_interfaces(self):
        pass

    def init_perception(self):
        self.chat_parser = NSPQuerier(self.opts)
        pass

    def init_controller(self):
        dialogue_object_classes = {}
        self.dialogue_manager = DialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            dialogue_object_mapper=DialogueObjectMapper,
            opts=self.opts
        )

TTAD_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "../models/semantic_parser/"
)
TTAD_BERT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../datasets/annotated_data/"
)
GROUND_TRUTH_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../datasets/ground_truth/"
)


class TestDialogueManager(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDialogueManager, self).__init__(*args, **kwargs)
        opts = MockOpt()
        opts.nsp_data_dir = TTAD_BERT_DATA_DIR
        opts.ground_truth_data_dir = GROUND_TRUTH_DATA_DIR
        opts.nsp_models_dir = TTAD_MODEL_DIR
        opts.no_ground_truth = False
        self.agent = FakeAgent(opts)

    def test_validate_bad_json(self):
        # Don't print debug info on failure since it will be misleading
        is_valid_json = self.agent.chat_parser.validate_parse_tree(parse_tree={}, debug=False)
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
                            "where_clause" : {
                                "AND": [{"pred_text": "has_name", "obj_text": [0, [5, 5]]}]
                            }
                        },
                    },
                }
            ],
        }
        is_valid_json = self.agent.chat_parser.validate_parse_tree(action_dict)
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
        is_valid_json = self.agent.chat_parser.validate_parse_tree(action_dict)
        self.assertTrue(is_valid_json)


if __name__ == "__main__":
    unittest.main()
