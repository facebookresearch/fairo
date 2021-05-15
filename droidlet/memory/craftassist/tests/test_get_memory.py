"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import droidlet.perception.craftassist.shapes as shapes
from craftassist.test.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.interpreter.tests.all_test_commands import *


class GetMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        cube_triples = {"has_name": "cube", "has_shape": "cube"}
        self.cube = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, -2), relations=cube_triples)
        sphere_triples = {"has_name": "sphere", "has_shape": "sphere"}
        self.sphere = self.add_object(
            shapes.sphere(radius=1), (11, 64, 2), relations=sphere_triples
        )
        triangle_triples = {"has_name": "triangle", "has_shape": "triangle"}
        self.triangle = self.add_object(shapes.triangle(), (6, 64, -5), relations=triangle_triples)
        self.set_looking_at(list(self.cube.blocks.keys())[0])

    def test_get_name_and_left_of(self):
        # set the name
        name = "fluffball"
        self.agent.memory.add_triple(subj=self.cube.memid, pred_text="has_name", obj_text=name)

        # get the name
        d = GET_MEMORY_COMMANDS["what is where I am looking"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn(name, self.last_outgoing_chat())

        d = GET_MEMORY_COMMANDS["what is to the left of the cube?"]
        self.handle_logical_form(d, stop_on_chat=True)
        # check that proper chat was sent
        self.assertIn("triangle", self.last_outgoing_chat())

        d = GET_MEMORY_COMMANDS["what is the thing closest to you?"]
        self.handle_logical_form(d, stop_on_chat=True)
        # check that proper chat was sent
        # note: the agent excludes itself from these by default, maybe fix?
        self.assertIn("SPEAKER", self.last_outgoing_chat())

        d = GET_MEMORY_COMMANDS["what is the thing closest to me?"]
        self.handle_logical_form(d, stop_on_chat=True)
        # check that proper chat was sent
        # note: the agent does NOT!! exclude SPEAKER...
        # FIXME?
        self.assertIn("SPEAKER", self.last_outgoing_chat())

    def test_what_are_you_doing(self):
        d = DANCE_COMMANDS["dance"]
        self.handle_logical_form(d)

        # start building a cube
        d = BUILD_COMMANDS["build a small cube"]
        self.handle_logical_form(d, max_steps=9)

        # what are you doing?
        d = GET_MEMORY_COMMANDS["what are you doing"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("build", self.last_outgoing_chat())
        assert not "dance" in self.last_outgoing_chat()

    def test_what_are_you_building(self):
        # start building a cube
        d = BUILD_COMMANDS["build a small cube"]
        self.handle_logical_form(d, max_steps=12)

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
        self.assertIn("42", self.last_outgoing_chat())
        self.assertIn("65", self.last_outgoing_chat())

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


class GetMemoryCountAndSizeTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        red_cube_triples = {"has_name": "cube", "has_shape": "cube", "has_colour": "red"}
        blue_cube_triples = {"has_name": "cube", "has_shape": "cube", "has_colour": "blue"}
        red_sphere_triples = {"has_name": "sphere", "has_shape": "sphere", "has_colour": "red"}
        blue_sphere_triples = {"has_name": "sphere", "has_shape": "sphere", "has_colour": "blue"}
        self.cube1 = self.add_object(
            shapes.cube(size=2, bid=(35, 14)), (19, 63, 14), relations=red_cube_triples
        )
        self.cube2 = self.add_object(
            shapes.cube(size=2, bid=(35, 14)), (15, 63, 15), relations=red_cube_triples
        )
        self.cube3 = self.add_object(
            shapes.cube(size=3, bid=(35, 11)), (14, 63, 19), relations=blue_cube_triples
        )
        self.sphere1 = self.add_object(
            shapes.sphere(bid=(35, 14), radius=2), (14, 63, 8), relations=red_sphere_triples
        )
        self.sphere2 = self.add_object(
            shapes.sphere(bid=(35, 11), radius=2), (8, 63, 14), relations=blue_sphere_triples
        )
        self.set_looking_at(list(self.cube1.blocks.keys())[0])

    def test_counts_and_size(self):
        d = GET_MEMORY_COMMANDS["how many cubes are there?"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("3", self.last_outgoing_chat())

        # Note that the command is slightly malformed, no "memory_type" key
        d = GET_MEMORY_COMMANDS["how many blue things are there?"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("2", self.last_outgoing_chat())

        d = GET_MEMORY_COMMANDS["how many blocks are in the blue cube?"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("27", self.last_outgoing_chat())

        d = GET_MEMORY_COMMANDS["how tall is the blue cube?"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("3", self.last_outgoing_chat())

        d = GET_MEMORY_COMMANDS["how wide is the red cube?"]
        self.handle_logical_form(d, stop_on_chat=True)

        # check that proper chat was sent
        self.assertIn("2", self.last_outgoing_chat())


if __name__ == "__main__":
    unittest.main()
