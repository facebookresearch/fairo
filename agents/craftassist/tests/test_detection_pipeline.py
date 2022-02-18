"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import numpy as np
import torch
from droidlet.base_util import euclid_dist
from droidlet.perception.craftassist.detection_model_perception import DetectionWrapper
from droidlet.interpreter.tests.all_test_commands import *
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from agents.craftassist.craftassist_agent import CraftAssistAgent


def fake_instseg_model(text_spans, blocks):
    masks = []
    blocks = torch.Tensor(blocks)
    for t in text_spans:
        mask = torch.zeros(blocks.shape[0], blocks.shape[1], blocks.shape[2])
        if "gold" in t:
            mask[torch.nonzero(blocks[:, :, :, 0] == 41, as_tuple=True)] = 1
        else:
            mask[torch.nonzero(blocks[:, :, :, 0] == 42, as_tuple=True)] = 1
        masks.append(mask)
    return masks


class ObjectsTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        # add two objects
        self.obj_a = self.add_object([((0, 63, z), (41, 0)) for z in [2, 3, 4]])
        self.obj_b = self.add_object([((x, 63, 0), (42, 0)) for x in [-4, -5]])
        self.detection_wrapper = DetectionWrapper(model=fake_instseg_model)

    def test_move_gold(self):
        d = MOVE_COMMANDS["go to the gold rectanguloid"]
        spans = self.agent.memory.nodes["Program"].get_refobj_text_spans(lf=d)
        model_out = CraftAssistAgent.run_voxel_model(self.agent, self.detection_wrapper, spans)
        self.agent.memory.update(model_out)

        self.handle_logical_form(d)
        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, (0, 63, 3)), 2)

    def test_move_iron(self):
        d = MOVE_COMMANDS["go to the iron rectanguloid"]
        spans = self.agent.memory.nodes["Program"].get_refobj_text_spans(lf=d)
        model_out = CraftAssistAgent.run_voxel_model(self.agent, self.detection_wrapper, spans)
        self.agent.memory.update(model_out)

        self.handle_logical_form(d)
        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, (-4, 63, 0)), 2)


if __name__ == "__main__":
    unittest.main()
