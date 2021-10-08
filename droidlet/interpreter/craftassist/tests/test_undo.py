"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import time

import droidlet.base_util
import droidlet.lowlevel.minecraft.shape_util
import droidlet.lowlevel.minecraft.shapes
from droidlet.dialog.dialogue_task import AwaitResponse
from droidlet.interpreter.tests.all_test_commands import *
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase


class UndoTest(BaseCraftassistTestCase):
    def test_undo_destroy(self):
        tag = "fluffy"

        # Build something
        obj = self.add_object(droidlet.lowlevel.minecraft.shapes.cube(bid=(41, 0)), (0, 63, 0))
        self.set_looking_at(list(obj.blocks.keys())[0])

        # Tag it
        d = PUT_MEMORY_COMMANDS["that is fluffy"]
        self.handle_logical_form(d)
        self.assertIn(tag, obj.get_tags())

        # Destroy it
        d = DESTROY_COMMANDS["destroy where I am looking"]
        self.handle_logical_form(d)
        self.assertIsNone(self.agent.memory.get_block_object_by_xyz(list(obj.blocks.keys())[0]))

        _, taskmems = self.agent.memory.basic_search("SELECT MEMORY FROM Task WHERE running>0")
        assert not taskmems

        # Undo destroy (will ask confirmation)
        d = OTHER_COMMANDS["undo"]
        self.handle_logical_form(d, max_steps=20)
        _, taskmems = self.agent.memory.basic_search(
            "SELECT MEMORY FROM Task WHERE action_name=awaitresponse"
        )
        # is the undo waiting for a response?
        assert any([m.prio > 0 for m in taskmems])

        # confirm undo
        # TODO change tests to record different speakers to avoid the sleep?
        # time.sleep(0.02)
        print("here")
        self.add_incoming_chat("yes", self.speaker, add_to_memory=True)
        self.flush()

        # Check that block object has tag
        newobj = self.agent.memory.get_block_object_by_xyz(list(obj.blocks.keys())[0])
        self.assertIsNotNone(newobj)
        self.assertIn(tag, newobj.get_tags())


if __name__ == "__main__":
    unittest.main()
