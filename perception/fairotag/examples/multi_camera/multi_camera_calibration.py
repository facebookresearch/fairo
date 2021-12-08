from os import path

import numpy as np
import cv2
import sophus as sp

import fairotag as frt

if __name__ == "__main__":
    # Load images
    img_dir = "figs"

    img_files1 = [path.join(img_dir, f"Master_{i}.jpg") for i in range(0, 25)]
    img_files2 = [path.join(img_dir, f"Sub1_{i}.jpg") for i in range(0, 25)]
    img_files3 = [path.join(img_dir, f"Sub2_{i}.jpg") for i in range(0, 25)]

    img_ls1 = [cv2.imread(img_file) for img_file in img_files1]
    img_ls2 = [cv2.imread(img_file) for img_file in img_files2]
    img_ls3 = [cv2.imread(img_file) for img_file in img_files3]
    snapshots = [s for s in zip(img_ls1, img_ls2, img_ls3)]

    # Initialize camera
    camera1 = frt.CameraModule()
    camera2 = frt.CameraModule()
    camera3 = frt.CameraModule()
    cameras = [camera1, camera2, camera3]

    # Set intrinsics
    intrinsics = {
        "matrix": np.array([[613.0, 0.0, 640.0], [0.0, 613.0, 360.0], [0.0, 0.0, 1.0]]),
        "dist_coeffs": np.zeros([1, 5]),
    }
    camera1.set_intrinsics(**intrinsics)
    camera2.set_intrinsics(**intrinsics)
    camera3.set_intrinsics(**intrinsics)

    # Register markers
    for camera in cameras:
        for m_id in range(17):
            camera.register_marker_size(m_id, length=0.02625)

    # Initialize scene
    scene = frt.Scene()

    camera_names = ["Master", "Sub1", "Sub2"]
    for camera_name in camera_names:
        pose_in_frame = sp.SE3() if camera_name == "Master" else None
        scene.add_camera(camera_name, frame="world", pose_in_frame=pose_in_frame)

    scene.add_frame("board")
    for m_id in range(17):
        pose_in_frame = sp.SE3() if m_id == 0 else None
        scene.add_marker(m_id, frame="board", length=0.02625, pose_in_frame=pose_in_frame)

    # Calibrate
    detected_markers_ls = []

    for snapshot in snapshots:
        detected_markers = {
            camera_name: camera.detect_markers(img)
            for camera_name, camera, img in zip(camera_names, cameras, snapshot)
        }
        detected_markers_ls.append(detected_markers)
    scene.calibrate_extrinsics(detected_markers_ls)

    # Visualize
    scene.render_scene()

    """
    # Register markers on checkerboard (17 markers of length 21mm)
    for m_id in range(17):
        scene.register_marker_size(m_id, 0.02625)

    # Add snapshots
    for i, imgs in enumerate(snapshots):
        scene.add_snapshot(imgs)

    # Calibrate extrinsics
    scene.calibrate_extrinsics(verbosity=1)

    # Visualize
    scene.visualize_snapshot([img_ls1[0], img_ls2[0], img_ls3[0]])
    """
