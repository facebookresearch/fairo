from os import path

import numpy as np
import cv2
import sophus as sp

from arucoX import CameraModule, Scene

if __name__ == "__main__":
    # Load images
    img_dir = "figs"

    img_files1 = [path.join(img_dir, f"Master_{i}.jpg") for i in range(0, 25)]
    img_files2 = [path.join(img_dir, f"Sub1_{i}.jpg") for i in range(0, 25)]
    img_files3 = [path.join(img_dir, f"Sub2_{i}.jpg") for i in range(0, 25)]

    img_ls1 = [cv2.imread(img_file) for img_file in img_files1]
    img_ls2 = [cv2.imread(img_file) for img_file in img_files2]
    img_ls3 = [cv2.imread(img_file) for img_file in img_files3]
    snapshots = zip(img_ls1, img_ls2, img_ls3)

    # Initialize objects
    camera1 = CameraModule()
    camera2 = CameraModule()
    camera3 = CameraModule()
    scene = Scene([camera1, camera2, camera3])

    # Set intrinsics
    intrinsics = {
        "matrix": np.array([[613.0, 0.0, 640.0], [0.0, 613.0, 360.0], [0.0, 0.0, 1.0]]),
        "dist_coeffs": np.zeros([1, 5]),
    }
    camera1.set_intrinsics(**intrinsics)
    camera2.set_intrinsics(**intrinsics)
    camera3.set_intrinsics(**intrinsics)

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
