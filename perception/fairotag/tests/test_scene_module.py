import pytest

import numpy as np
import sophus as sp

import fairotag as frt

NUM_SAMPLES = 150


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(0)


@pytest.fixture
def setup_dict():
    """
    Generate data for two static cameras and two moving markers on same object

    0: camera A
    1: camera B
    2: marker A
    3: marker B
    Transform between 0 and 1 are fixed
    Transform between 2 and 3 are fixed
    """
    # Randomly generate 2 sets of objects in same frame
    # (Note: have 2 and 3 relatively close)
    t01 = sp.SE3.exp(np.random.randn(6))
    t23 = sp.SE3.exp(np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]) * np.random.randn(6))

    # Sample noise
    sample_hi = np.array([0.5, 0.5, 0.5, np.pi / 2, np.pi / 2, np.pi / 2])
    sample_lo = np.array([0.0, 0.0, 0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2])

    # Sample observations
    t02_samples = []
    t12_samples = []
    t13_samples = []
    for i in range(NUM_SAMPLES):
        # Move 2 to random pose (sampled from uniform)
        t2 = sp.SE3.exp(np.random.uniform(low=sample_lo, high=sample_hi))

        # Compute pose of 1 & 3
        t1 = t01
        t3 = t2 * t23

        # Compute observations (from 0 to 2 & from 1 to 3)
        t02_samples.append(t2)
        t12_samples.append(t1.inverse() * t2)
        t13_samples.append(t1.inverse() * t3)

    return {
        "t01": t01,
        "t23": t23,
        "t02_samples": t02_samples,
        "t12_samples": t12_samples,
        "t13_samples": t13_samples,
    }


def test_scene_calibration1(setup_dict):
    """
    Assume both cameras can see marker A.
    Tests calibration of two cameras in the same workspace.
    """
    scene = frt.Scene()

    # Setup scene (Model base as a camera and ee as a marker)
    scene.add_camera("0", pose_in_frame=sp.SE3())
    scene.add_camera("1")

    scene.add_frame("ee")
    scene.add_marker(2, frame="ee", pose_in_frame=sp.SE3())

    # Parse observation data
    for t02, t12 in zip(setup_dict["t02_samples"][::-1], setup_dict["t12_samples"][::-1]):
        detected_markers = {
            "0": [frt.MarkerInfo(id=2, pose=t02, corner=None, length=None)],
            "1": [frt.MarkerInfo(id=2, pose=t12, corner=None, length=None)],
        }
        scene.add_snapshot(detected_markers)

    # Calibrate extrinsics
    scene.calibrate_extrinsics()

    # Check results
    t01_inferred = scene.get_camera_info("1")["pose_in_frame"]
    t01_gt = setup_dict["t01"]
    print(t01_inferred.log() - t01_gt.log())
    assert np.allclose(t01_inferred.log(), t01_gt.log(), atol=1e-2)

    # Try state estimation
    t12_sample = setup_dict["t12_samples"][-1]
    detected_markers = {
        "1": [frt.MarkerInfo(id=2, pose=t12_sample, corner=None, length=None)],
    }
    scene.update_pose_estimations(detected_markers)

    t02_inferred = scene.get_marker_info(2)["pose"]
    t02_gt = setup_dict["t02_samples"][-1]
    assert np.allclose(t02_inferred.log(), t02_gt.log(), atol=1e-2)


def _test_scene_calibration2(setup_dict):
    """
    Assume only cameras A can see marker A, and only camera B can see marker B.
    Tests a problem similar to hand-eye calibration of manipulators, where
    camera A is equivalent to the robot base, and marker A is equivalent to the
    end-effector.

    TODO: Disabled for now. Bring back when hand-eye calibration is solved.
    """
    scene = frt.Scene()

    # Setup scene (Model base as a camera and ee as a marker)
    scene.add_camera("0", pose_in_frame=sp.SE3())
    scene.add_camera("1")

    scene.add_frame("ee")
    scene.add_marker(2, frame="ee", pose_in_frame=sp.SE3())
    scene.add_marker(3, frame="ee")

    # Parse observation data
    for t02, t13 in zip(setup_dict["t02_samples"], setup_dict["t13_samples"]):
        detected_markers = {
            "0": [frt.MarkerInfo(id=2, pose=t02, corner=None, length=None)],
            "1": [frt.MarkerInfo(id=3, pose=t13, corner=None, length=None)],
        }
        scene.add_snapshot(detected_markers)

    # Calibrate extrinsics
    scene.calibrate_extrinsics()

    # Check results
    t01_inferred = scene.get_camera_info("1")["pose_in_frame"]
    t23_inferred = scene.get_marker_info(3)["pose_in_frame"]
    t01_gt = setup_dict["t01"]
    t23_gt = setup_dict["t23"]
    print(t01_inferred.log() - t01_gt.log())
    print(t23_inferred.log() - t23_gt.log())
    assert np.allclose(t01_inferred.log(), t01_gt.log(), atol=1e-2)
    assert np.allclose(t23_inferred.log(), t23_gt.log(), atol=1e-2)
