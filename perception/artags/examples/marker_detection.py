import time

import numpy as np
import sophus as sp
import torch
import cv2
import pyrealsense2 as rs

import arucoX as ax

from grasping_example import RealSenseCamera

# Markers
TABLE_MARKER_ID = 2
TABLE_MARKER_LENGTH = 0.1
OBJECT_MARKER_ID = 3
OBJECT_MARKER_LENGTH = 0.05

# Calib params
NUM_CALIB_SAMPLES = 5
CALIB_SAMPLE_INTERVAL = 0.5


if __name__ == "__main__":
    # Initialize camera
    camera = RealSenseCamera()
    matrix, dist_coeffs = camera.get_intrinsics()

    # Initialize camera module & scene module
    c = ax.CameraModule()
    c.set_intrinsics(matrix=matrix, dist_coeffs=dist_coeffs)
    scene = ax.Scene(cameras=[c])

    # Register markers
    scene.register_marker_size(TABLE_MARKER_ID, TABLE_MARKER_LENGTH)
    scene.register_marker_size(OBJECT_MARKER_ID, OBJECT_MARKER_LENGTH)
    scene.set_origin_marker(TABLE_MARKER_ID)

    #######################
    # Calibrate scene
    #######################
    # Capture images
    for _ in range(NUM_CALIB_SAMPLES):
        img = camera.get_image()
        scene.add_snapshot([img])
        time.sleep(CALIB_SAMPLE_INTERVAL)

    # Calibrate
    scene.calibrate_extrinsics()

    #######################
    # Marker experiment
    #######################
    while True:
        # Pose estimation
        img = camera.get_image()
        # pose = c.estimate_marker_pose(img, marker_id=OBJECT_MARKER_ID)
        pose = scene.estimate_marker_pose([img], marker_id=OBJECT_MARKER_ID)

        # Print
        print(f"Object pose: {pose}")
