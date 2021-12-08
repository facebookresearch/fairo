from os import path
import pickle

import cv2
import numpy as np
import pytest

from arucoX import CameraModule

CALIB_IMGFILE_LIST = [
    "examples/single_camera/figs/charuco_1.jpg",
    "examples/single_camera/figs/charuco_2.jpg",
    "examples/single_camera/figs/charuco_3.jpg",
    "examples/single_camera/figs/charuco_4.jpg",
    "examples/single_camera/figs/charuco_5.jpg",
]

INPUT_IMGFILE = "examples/single_camera/figs/test_5x5.jpg"
MARKER_LENGTH = 0.05


@pytest.fixture
def intrinsics():
    camera0 = CameraModule()
    calib_img_list = [cv2.imread(f) for f in CALIB_IMGFILE_LIST]
    camera0.calibrate_camera(calib_img_list)
    return camera0.get_intrinsics()


@pytest.fixture
def output_dict(intrinsics):
    camera1 = CameraModule()
    camera1.set_intrinsics(intrinsics=intrinsics)

    # Detect markers in input image
    camera1.register_marker_size(0, MARKER_LENGTH)
    camera1.register_marker_size(3, MARKER_LENGTH)
    camera1.register_marker_size(4, MARKER_LENGTH)

    img = cv2.imread(INPUT_IMGFILE)
    markers = camera1.detect_markers(img)

    # Visualize results
    img_rend = camera1.render_markers(img, markers=markers)

    return {
        "markers": markers,
        "img_rend": img_rend,
    }


@pytest.fixture
def ref_data(intrinsics, output_dict):
    """
    Loads reference data
    Saves current data as reference data if reference data does not exist
    """
    data_path = path.join("tests/data", "pyexample_ref_data.pkl")

    if path.exists(data_path):
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {
            "intrinsics": intrinsics,
            "markers": output_dict["markers"],
            "img_rend": output_dict["img_rend"],
        }
        with open(data_path, "wb") as f:
            pickle.dump(data_dict, f)

    return data_dict


def test_calibration(intrinsics, ref_data):
    for field0, field1 in zip(intrinsics, ref_data["intrinsics"]):
        assert np.allclose(field0, field1, atol=1e-1)


def test_marker_id(output_dict, ref_data):
    for marker_id, marker_ref in zip(output_dict["markers"], ref_data["markers"]):
        assert marker_id.id == marker_ref.id
        assert np.allclose(marker_id.corner, marker_ref.corner, atol=1.0)


def test_pose_estimation(output_dict, ref_data):
    for marker_id, marker_ref in zip(output_dict["markers"], ref_data["markers"]):
        if marker_ref.length is not None:
            assert marker_id.length == marker_ref.length
            assert np.allclose(marker_id.pose.log(), marker_ref.pose.log(), atol=1e-3)
