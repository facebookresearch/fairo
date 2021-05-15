"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest

import droidlet.perception.craftassist.shapes as shapes
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.interpreter.tests.all_test_commands import *


class CorefResolveTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_destroy_it(self):
        # build a gold cube
        d = BUILD_COMMANDS["build a gold cube at 0 66 0"]
        changes = self.handle_logical_form(d)

        # assert cube was built
        self.assertGreater(len(changes), 0)
        self.assertEqual(set(changes.values()), set([(41, 0)]))
        cube_xyzs = set(changes.keys())

        # destroy it
        d = DESTROY_COMMANDS["destroy it"]
        changes = self.handle_logical_form(d, chatstr="destroy it")

        # assert cube was destroyed
        self.assertEqual(cube_xyzs, set(changes.keys()))
        self.assertEqual(set(changes.values()), {(0, 0)})


if __name__ == "__main__":
    unittest.main()
