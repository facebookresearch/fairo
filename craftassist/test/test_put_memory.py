"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest

import craftassist.agent.shapes as shapes
from .base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.interpreter.tests.all_test_commands import *


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.cube_right = self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4))
        self.cube_left = self.add_object(shapes.cube(), (9, 63, 10))
        self.set_looking_at(list(self.cube_right.blocks.keys())[0])

    def test_good_job(self):
        d = PUT_MEMORY_COMMANDS["good job"]
        self.handle_logical_form(d)

    def test_tag(self):
        d = PUT_MEMORY_COMMANDS["that is fluffy"]
        self.handle_logical_form(d)

        # destroy it
        d = DESTROY_COMMANDS["destroy the fluffy object"]
        changes = self.handle_logical_form(d, answer="yes")

        # ensure it was destroyed
        self.assertEqual(changes, {k: (0, 0) for k in self.cube_right.blocks.keys()})

    def test_tag_and_build(self):
        d = PUT_MEMORY_COMMANDS["that is fluffy"]
        self.handle_logical_form(d)

        # build a fluffy
        d = BUILD_COMMANDS["build a fluffy here"]
        changes = self.handle_logical_form(d, answer="yes")

        self.assert_schematics_equal(changes.items(), self.cube_right.blocks.items())


if __name__ == "__main__":
    unittest.main()
