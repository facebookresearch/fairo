"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

import droidlet.base_util
import droidlet.lowlevel.minecraft.shape_helpers
import droidlet.lowlevel.minecraft.shapes
from droidlet.lowlevel.minecraft.mc_util import euclid_dist
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.shared_data_structs import MockOpt

TTAD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/semantic_parser/")
TTAD_BERT_DATA_DIR = os.path.join(os.path.dirname(__file__), "../datasets/annotated_data/")


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        opts = MockOpt()
        opts.nsp_models_dir = TTAD_MODEL_DIR
        opts.nsp_data_dir = TTAD_BERT_DATA_DIR
        super().setUp(agent_opts=opts)

        self.cube_right = self.add_object(droidlet.lowlevel.minecraft.shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(droidlet.lowlevel.minecraft.shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_come_here(self):
        chat = "come here"
        self.add_incoming_chat(chat, self.speaker)
        # get logical form
        preprocessed_chat, chat_parse = self.agent.perception_modules["language_understanding"].get_parse(chat)
        chat_memid = self.agent.memory.add_chat(self.agent.memory.get_player_by_name(self.speaker).memid,
                                                preprocessed_chat)
        logical_form_memid = self.agent.memory.add_logical_form(chat_parse)
        self.agent.memory.add_triple(subj=chat_memid, pred_text="has_logical_form", obj=logical_form_memid)
        self.agent.memory.tag(subj_memid=chat_memid, tag_text="unprocessed")
        self.flush()
        self.agent.memory.untag(subj_memid=chat_memid, tag_text="unprocessed")
        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_stop(self):
        chat = "stop"
        self.add_incoming_chat(chat, self.speaker)
        # get logical form
        preprocessed_chat, chat_parse = self.agent.perception_modules["language_understanding"].get_parse(chat)
        chat_memid = self.agent.memory.add_chat(self.agent.memory.get_player_by_name(self.speaker).memid,
                                                preprocessed_chat)
        logical_form_memid = self.agent.memory.add_logical_form(chat_parse)
        self.agent.memory.add_triple(subj=chat_memid, pred_text="has_logical_form", obj=logical_form_memid)
        self.agent.memory.tag(subj_memid=chat_memid, tag_text="unprocessed")
        self.flush()
        self.agent.memory.untag(subj_memid=chat_memid, tag_text="unprocessed")



if __name__ == "__main__":
    unittest.main()
