"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import os
import Pyro4
import sys

import cv2
import math
import numpy as np
from numpy.testing import assert_allclose

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
IP = "127.0.0.1"
if os.getenv("LOCOBOT_IP"):
    IP = os.getenv("LOCOBOT_IP")


def get_asset_path(name):
    """Returns the filename concatenated with the path to the test assets
    folder."""
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_assets", "expect", name)


def assert_image(image, expected_image_path):
    if os.getenv("EXPECTTEST_ACCEPT"):
        cv2.imwrite(expected_image_path, image)
    expected = cv2.imread(expected_image_path, cv2.IMREAD_UNCHANGED)
    assert expected is not None
    assert_allclose(image, expected)


def assert_visual(bot, key):
    assert type(key) == str
    image = bot.get_rgb()
    assert_image(image, get_asset_path(key + ".png"))


def assert_turn_degree(initial, final, degree):
    final_deg = math.degrees(initial[2]) + degree
    gt_final = (initial[0], initial[1], math.radians(final_deg))
    assert_allclose(gt_final, final, rtol=1e-5)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        global IP
        self.bot = Pyro4.Proxy("PYRONAME:remotelocobot@" + IP)
        self.nav = Pyro4.Proxy("PYRONAME:navigation@" + IP)

        if not hasattr(self, "initial_state"):
            self.initial_state = self.bot.get_base_state()
        else:
            # make sure after every unit test to go back to initial position
            self.bot.go_to_absolute(self.initial_state)

    def test_go_to_absolute(self):
        initial_state = [4.8, 0.16, -1.0]  # in apartment_0, right in front of the humans
        self.nav.go_to_absolute(initial_state)
        self.nav.go_to_absolute(initial_state)
        assert_allclose(initial_state, self.bot.get_base_state(), rtol=1e-3)
        # assert_visual(self.bot, "go_to_absolute1")

        for i in range(10):
            # test that multiple calls don't create side-effects
            self.nav.go_to_absolute(initial_state)
        assert_allclose(initial_state, self.bot.get_base_state(), rtol=1e-3)
        # assert_visual(self.bot, "go_to_absolute2")

    def test_turn(self):
        # turn a set of a angles
        turns_in_degrees = [0, 45, 90, -90, 180]
        for x in turns_in_degrees:
            self.nav.go_to_absolute([4.8, 0.16, -1.0])
            init = self.bot.get_base_state()
            self.nav.go_to_relative([0, 0, math.radians(x)])
            final = self.bot.get_base_state()
            assert_turn_degree(init, final, x)


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        global IP
        self.bot = Pyro4.Proxy("PYRONAME:remotelocobot@" + IP)
        self.nav = Pyro4.Proxy("PYRONAME:navigation@" + IP)
        initial_state = [4.8, 0.16, -1.0]  # in apartment_0, right in front of the humans
        # make sure after every unit test to go back to initial position
        self.nav.go_to_absolute(initial_state)
        self.nav.go_to_absolute(initial_state)

    def test_facerec(self):
        # fill this up
        pass


if __name__ == "__main__":
    unittest.main()
