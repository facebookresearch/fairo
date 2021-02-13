"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .core import AbstractHandler
from norfair import (
    Detection as NorfairDetection,
    Tracker,
    draw_tracked_objects,
    print_objects_as_table,
)
import numpy as np
import tempfile
from datetime import datetime
import cv2
import logging
import os


class TrackingHandler(AbstractHandler):
    """Class for real-time 2D object tracking.

    We use the results of DetectionHandler and the norfair tracking library (https://github.com/tryolabs/norfair)
    """

    def __init__(self):
        def euclidean_distance(detection, tracked_object):
            return np.linalg.norm(detection.points - tracked_object.estimate)

        self.tracker = Tracker(
            distance_function=euclidean_distance,
            distance_threshold=20,
            hit_inertia_max=25,
            point_transience=4,
        )

    def to_norfair(self, detections):
        d = []
        for x in detections:
            d.append(NorfairDetection(points=np.asarray([x.center])))
        return d

    def handle(self, rgb_depth, detections):
        """run tracker on the current rgb_depth frame for the detections found"""
        logging.info("In TrackingHandlerNorfair ... ")
        detections = self.to_norfair(detections)
        self.tracked_objects = self.tracker.update(detections, period=4)
        img = rgb_depth.rgb
        print_objects_as_table(self.tracked_objects)
        if os.getenv("DEBUG_DRAW") == "True":
            draw_tracked_objects(img, self.tracked_objects)
            cv2.imshow("Norfair", img)
            tf = tempfile.NamedTemporaryFile(
                prefix="norfair_" + str(datetime.now()) + "_locobot_capture_" + "_",
                suffix=".jpg",
                dir=".",
                delete=False,
            )
            cv2.imwrite(tf.name, img)
            cv2.waitKey(3)
