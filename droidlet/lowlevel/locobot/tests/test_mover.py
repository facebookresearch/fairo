"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import os
import sys
from .test_utils import assert_distance_moved, assert_turn_degree
from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover

IP = "127.0.0.1"
if os.getenv("LOCOBOT_IP"):
    IP = os.getenv("LOCOBOT_IP")


class MoverTests(unittest.TestCase):
    def setUp(self):
        # Good starting global pos to not casue collisons
        INIT = (4.8, 0.16, 0)
        self.agent = LoCoBotMover(ip=IP, backend="habitat")
        self.agent.bot.go_to_absolute(INIT, close_loop=False)

    def test_move_relative(self):
        for task_pos in [(0, 0, 0), (0, -0.1, 1.0), (0, 0.1, 0), (-0.1, -0.1, -1.0)]:
            initial_state = self.agent.bot.get_base_state(state_type="odom")
            self.agent.move_relative([task_pos])
            assert_distance_moved(
                initial_state, self.agent.bot.get_base_state(state_type="odom"), task_pos
            )

    def test_turn(self):
        # turn a set of a angles
        turns_in_degrees = [0, 45, 90, -90, 180]
        for x in turns_in_degrees:
            init = self.agent.bot.get_base_state(state_type="odom")
            self.agent.turn(x)
            final = self.agent.bot.get_base_state(state_type="odom")
            assert_turn_degree(init[2], final[2], x)


if __name__ == "__main__":
    unittest.main()
