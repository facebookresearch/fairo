"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from numpy.testing import assert_allclose
from numpy.linalg import norm
from numpy import array
import math
import unittest
import logging


def assert_distance_moved(pos1, pos2, movement_vector):
    act_dist = norm(array(pos1)[:2] - array(pos2)[:2])
    goal_dist = norm(array(movement_vector)[:2])
    assert_allclose(act_dist, goal_dist, rtol=1e-3)


def assert_turn_degree(initial_yaw, final_yaw, turn_degree):
    """Asserts the difference in final and initial yaws is degree."""
    logging.info(
        "{}, {}, {}".format(math.degrees(initial_yaw), math.degrees(final_yaw), turn_degree)
    )
    initial_d = math.degrees(initial_yaw) % 360
    final_d = math.degrees(final_yaw) % 360
    expect_fd = (initial_d + turn_degree) % 360
    # to make dist(0, 359.9) = dist(0, 0.1), convert angle to 2D location based on unit circle and then compare location
    final_loc = [math.cos(math.radians(final_d)), math.sin(math.radians(final_d))]
    expect_loc = [math.cos(math.radians(expect_fd)), math.sin(math.radians(expect_fd))]
    assert_allclose(final_loc, expect_loc, atol=2e-7)


class UtilsTest(unittest.TestCase):
    def test_assert_turn_degree(self):
        self.assertRaises(AssertionError, assert_turn_degree, 0, math.radians(10), 90)
        self.assertRaises(AssertionError, assert_turn_degree, 0, 0, 90)
        self.assertRaises(AssertionError, assert_turn_degree, 0, 0, -90)
        self.assertRaises(AssertionError, assert_turn_degree, 0, 0, 180)
        assert_turn_degree(0, 0, 360)
        assert_turn_degree(0, math.radians(90), 90)
        assert_turn_degree(math.radians(90), math.radians(180), 90)
        assert_turn_degree(math.radians(-45), math.radians(45), 90)
        assert_turn_degree(math.radians(45), math.radians(-45), -90)
        assert_turn_degree(math.radians(-45), math.radians(-45), 360)


if __name__ == "__main__":
    unittest.main()
