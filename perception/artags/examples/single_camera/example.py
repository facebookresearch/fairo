import numpy as np
import cv2

import arucoX as ax

CALIB_IMGFILE_LIST = [
    "figs/charuco_1.jpg",
    "figs/charuco_2.jpg",
    "figs/charuco_3.jpg",
    "figs/charuco_4.jpg",
    "figs/charuco_5.jpg",
]

INPUT_IMGFILE = "figs/test_5x5.jpg"
MARKER_LENGTH = 0.05


def main():
    c = ax.CameraModule()

    # Calibrate camera
    calib_img_list = [cv2.imread(f) for f in CALIB_IMGFILE_LIST]
    c.calibrate_camera(calib_img_list)

    # Register half of the markers
    c.register_marker_size(0, MARKER_LENGTH)
    c.register_marker_size(3, MARKER_LENGTH)
    c.register_marker_size(4, MARKER_LENGTH)

    # Detect markers in input image
    img = cv2.imread(INPUT_IMGFILE)
    markers = c.detect_markers(img)

    # Visualize results
    img_rend = c.render_markers(img, markers=markers)
    cv2.imshow("output_img", img_rend)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
