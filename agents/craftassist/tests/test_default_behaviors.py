import unittest
import numpy as np
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.interpreter.craftassist.default_behaviors import build_random_shape, come_to_player
from droidlet.lowlevel.minecraft import shape_util as sh

class TestDefaultBehavior(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()

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
