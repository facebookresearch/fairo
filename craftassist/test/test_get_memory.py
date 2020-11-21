"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from base_agent.dialogue_objects import SPEAKERLOOK
import craftassist.agent.shapes as shapes
from base_craftassist_test_case import BaseCraftassistTestCase
from all_test_commands import *


class GetMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_get_name(self):
        # set the name
        name = "fluffball"
        self.agent.memory.add_triple(
            subj=self.cube_right.memid, pred_text="has_name", obj_text=name
        )

        # get the name
        d = GET_MEMORY_COMMANDS["what is where I am looking"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn(name, self.last_outgoing_chat())

    def test_what_are_you_doing(self):
        # start building a cube
        d = BUILD_COMMANDS["build a small cube"]
        self.handle_logical_form(d, max_steps=5)

        # what are you doing?
        d = GET_MEMORY_COMMANDS["what are you doing"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("building", self.last_outgoing_chat())

    def test_what_are_you_building(self):
        # start building a cube
        d = BUILD_COMMANDS["build a small cube"]
        self.handle_logical_form(d, max_steps=5)

        # what are you building
        d = GET_MEMORY_COMMANDS["what are you building"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("cube", self.last_outgoing_chat())

    def test_where_are_you_going(self):
        # start moving
        d = MOVE_COMMANDS["move to 42 65 0"]
        self.handle_logical_form(d, max_steps=3)

        # where are you going?
        d = GET_MEMORY_COMMANDS["where are you going"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("(42, 65, 0)", self.last_outgoing_chat())

    def test_where_are_you(self):
        # move to origin
        d = MOVE_COMMANDS["move to 0 63 0"]
        self.handle_logical_form(d)

        # where are you?
        d = GET_MEMORY_COMMANDS["where are you"]
        self.handle_logical_form(d)

        # check that proper chat was sent
        loc_in_chat = (
            "(0.0, 63.0, 0.0)" in self.last_outgoing_chat()
            or "(0, 63, 0)" in self.last_outgoing_chat()
        )
        assert loc_in_chat


#        self.assertIn("(0.0, 63.0, 0.0)", self.last_outgoing_chat())


if __name__ == "__main__":
    unittest.main()
