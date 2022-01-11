from os import path
import json

import cv2
import numpy as np
import pytest

import fairotag as frt

CALIB_IMGFILE_LIST = [
    "tutorials/data/0_marker_pose_estimation/charuco_1.jpg",
    "tutorials/data/0_marker_pose_estimation/charuco_2.jpg",
    "tutorials/data/0_marker_pose_estimation/charuco_3.jpg",
    "tutorials/data/0_marker_pose_estimation/charuco_4.jpg",
    "tutorials/data/0_marker_pose_estimation/charuco_5.jpg",
]

INPUT_IMGFILE = "tutorials/data/0_marker_pose_estimation/test_5x5.jpg"
MARKER_LENGTH = 0.05


@pytest.fixture
def intrinsics():
    camera0 = frt.CameraModule()
    calib_img_list = [cv2.imread(f) for f in CALIB_IMGFILE_LIST]
    camera0.calibrate_camera(calib_img_list)
    return camera0.get_intrinsics()


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


@pytest.fixture
def ref_data(intrinsics, detected_markers):
    """
    Loads reference data
    Saves current data as reference data if reference data does not exist
    """
    data_path = path.join("tests/data", "camera_module_ref_data.json")

    if path.exists(data_path):
        with open(data_path, "r") as f:
            data_dict = json.load(f)
    else:
        data_dict = {}
        data_dict["intrinsics"] = frt.utils.intrinsics2dict(intrinsics)
        data_dict["markers"] = {}
        for m in detected_markers:
            id = int(m.id)
            data_dict["markers"][id] = {}
            data_dict["markers"][id]["corner"] = m.corner.tolist()
            if m.pose is not None:
                data_dict["markers"][id]["pose"] = m.pose.log().tolist()

        with open(data_path, "w") as f:
            json.dump(data_dict, f)

    return data_dict


def test_calibration(intrinsics, ref_data):
    for field0, field1 in zip(intrinsics, frt.utils.dict2intrinsics(ref_data["intrinsics"])):
        assert np.allclose(field0, field1, atol=1e-1)


def test_marker_id(detected_markers, ref_data):
    for marker_out in detected_markers:
        assert str(marker_out.id) in ref_data["markers"]
        assert np.allclose(
            marker_out.corner, ref_data["markers"][str(marker_out.id)]["corner"], atol=1e0
        )


def test_pose_estimation(detected_markers, ref_data):
    for marker_out in detected_markers:
        if marker_out.length is not None:
            assert np.allclose(
                marker_out.pose.log(), ref_data["markers"][str(marker_out.id)]["pose"], atol=1e-3
            )
