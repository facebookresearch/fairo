"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

import droidlet.perception.craftassist.shapes as shapes
from droidlet.lowlevel.minecraft.mc_util import euclid_dist
from craftassist.test.base_craftassist_test_case import BaseCraftassistTestCase
from craftassist.test.fake_agent import MockOpt


TTAD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../../../craftassist/agent/models/semantic_parser/")
TTAD_BERT_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../craftassist/agent/datasets/annotated_data/")


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        opts = MockOpt()
        opts.nsp_models_dir = TTAD_MODEL_DIR
        opts.nsp_data_dir = TTAD_BERT_DATA_DIR
        super().setUp(agent_opts=opts)

        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_come_here(self):
        chat = "come here"
        self.add_incoming_chat(chat, self.speaker)
        self.flush()

        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_stop(self):
        chat = "stop"
        self.add_incoming_chat(chat, self.speaker)
        self.flush()


if __name__ == "__main__":
    unittest.main()
