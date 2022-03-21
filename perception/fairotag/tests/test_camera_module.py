from os import path
import json
from enum import Enum

import cv2
import numpy as np
import sophus as sp
import pytest

import fairotag as frt

CALIB_IMGFILE_LIST = [
    "tutorials/data/1_marker_pose_estimation/charuco_1.jpg",
    "tutorials/data/1_marker_pose_estimation/charuco_2.jpg",
    "tutorials/data/1_marker_pose_estimation/charuco_3.jpg",
    "tutorials/data/1_marker_pose_estimation/charuco_4.jpg",
    "tutorials/data/1_marker_pose_estimation/charuco_5.jpg",
]

INPUT_IMGFILE = "tutorials/data/1_marker_pose_estimation/test_5x5.jpg"
MARKER_LENGTH = 0.05

INTRINSICS_FILE = "tutorials/data/realsense_intrinsics.json"


class IntrinsicsType(Enum):
    FROM_CALIB = 1
    FROM_FILE = 2


@pytest.fixture(params=[IntrinsicsType.FROM_FILE, IntrinsicsType.FROM_CALIB])
def intrinsics(request):
    if request.param == IntrinsicsType.FROM_CALIB:
        camera0 = frt.CameraModule()
        calib_img_list = [cv2.imread(f) for f in CALIB_IMGFILE_LIST]
        camera0.calibrate_camera(calib_img_list)
        return camera0.get_intrinsics()

    else:
        with open("tutorials/data/realsense_intrinsics.json", "r") as f:
            intrinsics = json.load(f)
        return frt.utils.dict2intrinsics(intrinsics)


@pytest.fixture
def detected_markers(intrinsics):
    camera1 = frt.CameraModule()
    camera1.set_intrinsics(intrinsics=intrinsics)

    # Detect markers in input image
    camera1.register_marker_size(0, MARKER_LENGTH)
    camera1.register_marker_size(3, MARKER_LENGTH)
    camera1.register_marker_size(4, MARKER_LENGTH)

    img = cv2.imread(INPUT_IMGFILE)
    markers = camera1.detect_markers(img)

    return markers


def test_detected_markers(detected_markers):
    marker_ids = [0, 1, 2, 3, 4, 5]
    registered_marker_ids = [0, 3, 4]

    for marker in detected_markers:
        assert marker.id in marker_ids

        if marker.id in registered_marker_ids:
            assert marker.length == MARKER_LENGTH
            assert type(marker.pose) is sp.SE3
        else:
            assert marker.length is None
            assert marker.pose is None
