"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .core import AbstractHandler
import cv2
import os
import numpy as np


class InputHandler(AbstractHandler):
    """Fetches all input sensor data from the robot.

    Args:
        agent (LocoMCAgent): reference to the agent instance
        read_from_camera (boolean): boolean to enable/disable reading from the camera
            sensor and possibly reading from a local video instead (default: True)
    """

    def __init__(self, agent, read_from_camera=True):
        self.agent = agent
        self.read_from_camera = read_from_camera

    def handle(self):
        if self.read_from_camera:
            rgb_depth = self.agent.mover.get_rgb_depth()
        else:
            # implement reading from a file on disk for quicker iteration.
            raise NotImplementedError
        if os.getenv("DEBUG_DRAW") == "True":
            self._debug_draw(rgb_depth)
        return rgb_depth

    def _debug_draw(self, rgb_depth):
        pil_img = rgb_depth.get_pillow_image()
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Insight", img[:, :, ::-1])
        cv2.waitKey(3)
