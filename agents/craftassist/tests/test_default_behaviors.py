import unittest
import numpy as np
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.interpreter.craftassist.default_behaviors import build_random_shape, come_to_player


class TestDefaultBehavior(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()

    def test_build_random_shape(self):
        schematic = build_random_shape(self.agent)
        # Assert that some non-zero size schematic was built
        self.assertTrue(len(schematic) > 0)
        changes = self.flush(500)
        # Assert that build task was pushed and completed successfully
        self.assertTrue(self.agent.memory.get_last_finished_root_task().action_name, "build")
        self.assertTrue(len(changes) == len(schematic))

    def test_come_to_player(self):
        come_to_player(self.agent)
        changes = self.flush(30)
        # Assert that a move task was pushed and finished
        self.assertTrue(self.agent.memory.get_last_finished_root_task().action_name, "move")
        # Assert that agent moved to player
        distance = np.linalg.norm(np.array(self.agent.get_other_players()[0].pos) - self.agent.pos)
        self.assertTrue(distance < 3.5)


if __name__ == "__main__":
    unittest.main()
