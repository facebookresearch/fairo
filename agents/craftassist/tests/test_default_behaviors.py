import unittest
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.interpreter.craftassist.default_behaviors import build_random_shape, come_to_player


class TestDefaultBehavior(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()

    def test_build_random_shape(self):
        schematic = build_random_shape(self.agent)
        # Assert that some non-zero size schematic was built
        self.assertTrue(len(schematic) > 0)
        # Assert that build task was pushed and completed successfully
        self.assertTrue(self.agent.memory.get_last_finished_root_task().action_name, "build")

    def test_come_to_player(self):
        come_to_player(self.agent)
        # Assert that a move task was pushed and executed succesfully
        self.assertTrue(self.agent.memory.get_last_finished_root_task().action_name, 'move')


if __name__ == "__main__":
    unittest.main()
