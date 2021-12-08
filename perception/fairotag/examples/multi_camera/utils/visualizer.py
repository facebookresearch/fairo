import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class Scene:
    def __init__(self, camera_size=0.1):
        self.camera_edge = 0.5 * camera_size

        # Initialize plot
        self.fig = plt.figure()
        self.ax = plt.axes(projection="3d")

    def add_camera(self, pose, color="k"):
        # Draw a pyramid representing a camera
        b0_0 = self.camera_edge * np.array([1, 1, 2])
        b1_0 = self.camera_edge * np.array([1, -1, 2])
        b2_0 = self.camera_edge * np.array([-1, -1, 2])
        b3_0 = self.camera_edge * np.array([-1, 1, 2])
        t_0 = np.zeros(3)

        b0 = pose * b0_0
        b1 = pose * b1_0
        b2 = pose * b2_0
        b3 = pose * b3_0
        t = pose * t_0

        starts = [b0, b1, b2, b3, b0, b1, b2, b3]
        ends = [b1, b2, b3, b0, t, t, t, t]
        for s, e in zip(starts, ends):
            self.frt.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color)

    def add_marker(self, pose, color="k"):
        # Draw a dot representing a marker
        x, y, z = pose.translation()
        self.frt.scatter(x, y, z, color=color)

    def draw(self):
        plt.show()
