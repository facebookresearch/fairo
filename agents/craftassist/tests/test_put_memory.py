"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest

import droidlet.base_util
import droidlet.lowlevel.minecraft.shape_helpers
import droidlet.lowlevel.minecraft.shapes
from droidlet.memory.memory_nodes import PlayerNode, SetNode
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.interpreter.tests.all_test_commands import *
from droidlet.base_util import Player, Pos, Look


class PutMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.cube_right = self.add_object(
            droidlet.lowlevel.minecraft.shapes.cube(bid=(42, 0)), (9, 63, 4)
        )
        self.cube_left = self.add_object(droidlet.lowlevel.minecraft.shapes.cube(), (9, 63, 10))
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


class PutSetMemoryTestCase(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        memory = self.agent.memory
        PlayerNode.create(memory, Player(10, "agent10", Pos(1, 63, 1), Look(0, 0)))
        PlayerNode.create(memory, Player(11, "agent11", Pos(-2, 63, 1), Look(0, 0)))
        PlayerNode.create(memory, Player(12, "agent12", Pos(-4, 63, -3), Look(0, 0)))

        self.set_looking_at([1, 63, 1])

    def test_build_set(self):
        d = PUT_MEMORY_COMMANDS["you two are team alpha"]
        self.handle_logical_form(d)
        set_memids, _ = self.agent.memory.basic_search(
            "SELECT MEMORIES FROM Set WHERE (has_name=team alpha)"
        )
        assert len(set_memids) == 1
        set_memid = set_memids[0]
        agent_memids = self.agent.memory.basic_search(
            "SELECT MEMORIES FROM Player WHERE member_of=#={}".format(set_memid)
        )
        assert len(agent_memids) == 2


if __name__ == "__main__":
    unittest.main()
