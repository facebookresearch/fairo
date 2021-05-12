"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
import time
import droidlet.perception.craftassist.shapes as shapes
from droidlet.dialog.dialogue_objects import AwaitResponse
from droidlet.interpreter.tests.all_test_commands import *
from .base_craftassist_test_case import BaseCraftassistTestCase


class UndoTest(BaseCraftassistTestCase):
    def test_undo_destroy(self):
        tag = "fluffy"

        # Build something
        obj = self.add_object(shapes.cube(bid=(41, 0)), (0, 63, 0))
        self.set_looking_at(list(obj.blocks.keys())[0])

        # Tag it
        d = PUT_MEMORY_COMMANDS["that is fluffy"]
        self.handle_logical_form(d)
        self.assertIn(tag, obj.get_tags())

        # Destroy it
        d = DESTROY_COMMANDS["destroy where I am looking"]
        self.handle_logical_form(d)
        self.assertIsNone(self.agent.memory.get_block_object_by_xyz(list(obj.blocks.keys())[0]))

        # Undo destroy (will ask confirmation)
        d = OTHER_COMMANDS["undo"]
        self.handle_logical_form(d)
        self.assertIsInstance(self.agent.dialogue_manager.dialogue_stack.peek(), AwaitResponse)

        # confirm undo
        # TODO change tests to record different speakers to avoid the sleep?
        time.sleep(0.02)
        self.add_incoming_chat("yes", self.speaker)
        self.flush()

        # Check that block object has tag
        newobj = self.agent.memory.get_block_object_by_xyz(list(obj.blocks.keys())[0])
        self.assertIsNotNone(newobj)
        self.assertIn(tag, newobj.get_tags())


if __name__ == "__main__":
    unittest.main()
