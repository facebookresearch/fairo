"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .core import AbstractHandler
from .detector import Detection
import cv2
import numpy as np
import logging
import os


class DetectLaserPointer(AbstractHandler):
    """Identifies an especially prepared laser pointer in an image to use it
    for proprioception.
    """

    @staticmethod
    def get_hsv_laser_mask(image):
        """get the masked image with the laser dot using HSV color model."""
        # split the video frame into color channels
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 0, 255])
        upper_red = np.array([255, 255, 255])
        rough_mask = cv2.inRange(hsv_img, lower_red, upper_red)
        return rough_mask

    @staticmethod
    def get_laser_point_center(mask):
        """get the center of a masked laser pointer in screen coordinate."""
        center = None
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(contours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)
            if moments["m00"] > 0:
                center = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
            else:
                center = int(x), int(y)

            # only proceed if the radius meets a maximum size
            # 10 is approximately what would a laser dot would have
            if radius < 10:
                return center

    def forward(self, rgb_depth):
        logging.info("In LaserPointerHandler ... ")
        rgb = rgb_depth.rgb

        laser_mask = self.get_hsv_laser_mask(rgb)
        center = self.get_laser_point_center(laser_mask)
        if center:
            logging.debug(
                f"LASER POINTER is at: ({center[0]}, {center[1]}) in the screen coordinates (the image dimensions)"
            )

            if os.getenv("DEBUG_DRAW") == "True":
                # draw the image with a circle around the laser
                cv2.circle(rgb, center, 30, (0, 255, 255), 2)
                cv2.circle(rgb, center, 5, (0, 0, 255), -1)
                cv2.imshow("Track Laser", rgb)
                cv2.waitKey(3)

            return Detection(
                rgb_depth,
                class_label="iam point",  # same as ground_truth label
                properties=None,
                mask=laser_mask,
                bbox=None,
                center=center,
            )

        return None

    def _debug_draw(self, rgb_img, mask):
        cv2.imshow("mask", mask)
        cv2.imshow("Track Laser", rgb_img)
