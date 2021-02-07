"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import os
import Pyro4
import sys

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
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
        if not hasattr(self, "initial_state"):
            self.initial_state = self.bot.get_base_state(state_type="odom")
        else:
            # make sure after every unit test to go back to initial position
            self.bot.go_to_absolute(self.initial_state, close_loop=False)

    def test_go_to_absolute(self):
        initial_state = [4.8, 0.16, -1.0]  # in apartment_0, right in front of the humans
        self.bot.go_to_absolute(initial_state, close_loop=False)
        self.bot.go_to_absolute(initial_state, close_loop=False)
        assert_allclose(initial_state, self.bot.get_base_state("odom"), rtol=1e-3)
        # assert_visual(self.bot, "go_to_absolute1")

        for i in range(10):
            # test that multiple calls don't create side-effects
            self.bot.go_to_absolute(initial_state, close_loop=False)
        assert_allclose(initial_state, self.bot.get_base_state("odom"), rtol=1e-3)
        # assert_visual(self.bot, "go_to_absolute2")

    def test_turn(self):
        # turn a set of a angles
        turns_in_degrees = [0, 45, 90, -90, 180]
        for x in turns_in_degrees:
            self.bot.go_to_absolute([4.8, 0.16, -1.0], close_loop=False)
            init = self.bot.get_base_state(state_type="odom")
            self.bot.go_to_relative([0, 0, math.radians(x)], close_loop=False)
            final = self.bot.get_base_state(state_type="odom")
            assert_turn_degree(init, final, x)


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        global IP
        self.bot = Pyro4.Proxy("PYRONAME:remotelocobot@" + IP)
        initial_state = [4.8, 0.16, -1.0]  # in apartment_0, right in front of the humans
        # make sure after every unit test to go back to initial position
        self.bot.go_to_absolute(initial_state, close_loop=False)
        self.bot.go_to_absolute(initial_state, close_loop=False)

    def test_pix_to_3d(self):
        # test that it runs error-free
        self.bot.dip_pix_to_3dpt()

        # [y, x] points each in a particular quadrant
        pts_input = [[128, 384, 384, 128], [384, 384, 128, 128]]

        groundtruth_global = np.array(
            [
                [0.943435, -0.471717, 1.071717],
                [1.401783, -0.700892, -0.100892],
                [1.378382, 0.689191, -0.089191],
                [3.258884, 1.629442, 2.229442],
            ]
        )

        groundtruth_cam = np.array(
            [
                [0.4717174172401428, -0.4717174172401428, 0.9434348344802856],
                [0.7008916735649109, 0.7008916735649109, 1.4017833471298218],
                [-0.6891908049583435, 0.6891908049583435, 1.378381609916687],
                [-1.6294418573379517, -1.6294418573379517, 3.2588837146759033],
            ]
        )

        pts_global = self.bot.pix_to_3dpt(pts_input[0], pts_input[1])
        pts_in_cam = self.bot.pix_to_3dpt(pts_input[0], pts_input[1], in_cam=True)

        assert_allclose(pts_global[0], groundtruth_global, rtol=1e-2)
        assert_allclose(pts_in_cam[0], groundtruth_cam, rtol=1e-2)

    def test_pix_to_3d_shape(self):
        depth = self.bot.get_depth()
        h = depth.shape[0]
        w = depth.shape[1]
        rs = np.repeat(np.arange(h), w).ravel()
        cs = np.repeat(np.arange(w)[None, :], h, axis=0).ravel()
        pts_global, colors = self.bot.pix_to_3dpt(rs, cs)
        # assert that shape for both is h * w * 3 as per
        self.assertEqual(len(pts_global), h * w)
        self.assertEqual(len(colors), h * w)
        self.assertEqual(len(pts_global[0]), 3)
        self.assertEqual(len(colors[0]), 3)

    def test_facerec(self):
        # fill this up
        pass


if __name__ == "__main__":
    unittest.main()
