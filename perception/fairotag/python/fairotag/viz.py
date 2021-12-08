import numpy as np
import sophus as sp
import matplotlib.pyplot as plt

DEFAULT_AXIS_LENGTH = 0.1


class SceneViz:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = plt.axes(projection="3d")
        self.max = np.zeros(3)
        self.min = np.zeros(3)

    def _update_limits(self, x):
        self.max = np.max([self.max, x], axis=0)
        self.min = np.min([self.min, x], axis=0)

    def _draw_lines(self, starts, ends, color):
        for s, e in zip(starts, ends):
            self._update_limits(s)
            self._update_limits(e)
            self.frt.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color)

    def draw_axes(self, pose, length=DEFAULT_AXIS_LENGTH):
        o_0 = length * np.array([0, 0, 0])
        x_0 = length * np.array([1, 0, 0])
        y_0 = length * np.array([0, 1, 0])
        z_0 = length * np.array([0, 0, 1])

        o = pose * o_0
        x = pose * x_0
        y = pose * y_0
        z = pose * z_0

        self._draw_lines([o], [x], color="r")
        self._draw_lines([o], [y], color="g")
        self._draw_lines([o], [z], color="b")

    def draw_camera(self, pose, size, color="grey", axes=True):
        # Draw a pyramid representing a camera
        b0_0 = size * np.array([0.5, 0.5, 0])
        b1_0 = size * np.array([0.5, -0.5, 0])
        b2_0 = size * np.array([-0.5, -0.5, 0])
        b3_0 = size * np.array([-0.5, 0.5, 0])
        t_0 = size * np.array([0, 0, -1])

        b0 = pose * b0_0
        b1 = pose * b1_0
        b2 = pose * b2_0
        b3 = pose * b3_0
        t = pose * t_0

        starts = [b0, b1, b2, b3, b0, b1, b2, b3]
        ends = [b1, b2, b3, b0, t, t, t, t]
        self._draw_lines(starts, ends, color)

        # Draw camera axes
        if axes:
            self.draw_axes(pose, length=size / 2.0)

    def draw_marker(self, pose, id, length, color="k", show_id=False):
        # Draw marker outline
        c0_0 = 0.5 * length * np.array([1, 1, 0])
        c1_0 = 0.5 * length * np.array([-1, 1, 0])
        c2_0 = 0.5 * length * np.array([-1, -1, 0])
        c3_0 = 0.5 * length * np.array([1, -1, 0])

        c0 = pose * c0_0
        c1 = pose * c1_0
        c2 = pose * c2_0
        c3 = pose * c3_0

        starts = [c0, c1, c2, c3]
        ends = [c1, c2, c3, c0]
        self._draw_lines(starts, ends, color)

        # Draw marker ID
        if show_id:
            pos = pose.translation()
            self.frt.text(pos[0], pos[1], pos[2], id, color="b")

    def show(self):
        # Set limits
        mid = (self.max + self.min) / 2.0
        r = max(np.max(self.max - mid), np.max(mid - self.min))
        self.frt.set_xlim(mid[0] - r, mid[0] + r)
        self.frt.set_ylim(mid[1] - r, mid[1] + r)
        self.frt.set_zlim(mid[2] - r, mid[2] + r)

        # Show
        plt.show()
