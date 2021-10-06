import copy
import pickle
from collections import namedtuple

import cv2
import numpy as np
import sophus as sp

import matplotlib.pyplot as plt

from .camera import MarkerInfo
from .graph import FactorGraph


CAMERA_SIZE = 0.03
MARKER_OBS_SIG = [0.02, 0.02, 0.06, 0.1, 0.1, 0.1]
MARKER_TRANS_SIG = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]


CameraInfo = namedtuple("CameraInfo", "module pose")
SnapshotInfo = namedtuple("SnapshotInfo", "id camera_outputs")


class Scene:
    def __init__(self, cameras=[]):
        self.cameras = [CameraInfo(c, sp.SE3()) for c in cameras]

        self.snapshots = {}
        self.snapshot_counter = 0

        self.origin_id = None

        # Set up factor graph
        self.graph = FactorGraph()
        self.c_map = {}
        self.m_map = {}
        for i, c in enumerate(cameras):
            idx = self.graph.add_obj(is_static=True)
            self.c_map[i] = idx

    # Scene construction
    def add_snapshot(self, imgs):
        self._check_img_input(imgs)

        # Locate markers
        camera_outputs = [c.module.detect_markers(img) for c, img in zip(self.cameras, imgs)]

        # Register snapshot
        id = self.snapshot_counter
        snapshot_info = SnapshotInfo(id, camera_outputs)
        self.snapshots[id] = snapshot_info

        self.snapshot_counter += 1

        return id

    def remove_snapshot(self, id):
        self.snapshots.pop(id)

    def clear_snapshots(self):
        self.snapshots = {}

    # Scene query
    def get_camera_pose(self, id):
        assert id in range(len(self.cameras))
        return self.cameras[id].pose

    def get_num_cameras(self):
        return len(self.cameras)

    def get_num_snapshots(self):
        return len(self.snapshots)

    def get_snapshot(self, id):
        return self.snapshots[id]

    # Marker registration
    def register_marker(self, marker_id, length, static=False, pose=None):
        for c in self.cameras:
            c.module.register_marker_size(marker_id, length)

        # Add to factor graph
        idx = self.graph.add_obj(is_static=static)
        self.m_map[marker_id] = idx

        # Fix pose
        if pose is not None:
            self.graph.fix_obj_pose(idx, pose)

    # Marker detection & estimation within scene
    def reset_tracking(self):
        self.graph.reset()

    def track_markers(self, imgs):
        self._check_img_input(imgs)

        camera_outputs = [c.module.detect_markers(img) for c, img in zip(self.cameras, imgs)]
        markers = self._combine_marker_detections(camera_outputs)

        noise = MARKER_OBS_SIG
        trans_noise = MARKER_TRANS_SIG
        for m in markers:
            # Skip if not registered
            if m["id"] not in self.m_map.keys():
                continue

            # Add observation to graph
            pose_init = False
            for i, pose in enumerate(m["poses"]):
                if pose is not None:
                    self.graph.add_observation(
                        self.c_map[i], self.m_map[m["id"]], pose, noise, trans_noise
                    )
                    if not pose_init:
                        camera_pose = self.get_camera_pose(0)
                        self.graph.set_obj_pose(self.m_map[m["id"]], pose * camera_pose.inverse())
                        pose_init = True

        self.graph.increment()

        results = []
        for m in markers:
            if m["length"] is not None:
                marker_info = MarkerInfo(
                    m["id"], m["corner"], m["length"], self.graph.get_obj_pose(self.m_map[m["id"]])
                )
                results.append(marker_info)

        # Update camera extrinsics
        for i, idx in enumerate(self.c_map):
            pose = self.graph.get_obj_pose(idx)
            self.cameras[i] = self.cameras[i]._replace(pose=pose)

        return results

    def get_marker_pose(self, marker_id):
        return self.graph.get_obj_pose(self.m_map[marker_id])

    def detect_markers(self, imgs, fast=False):
        self._check_img_input(imgs)

        camera_outputs = [c.module.detect_markers(img) for c, img in zip(self.cameras, imgs)]
        markers = self._combine_marker_detections(camera_outputs)

        for m in markers:
            m["pose"] = self._filter_marker_poses(m["poses"], fast=fast)

        return [MarkerInfo(m["id"], m["corner"], m["length"], m["pose"]) for m in markers]

    def estimate_marker_pose(self, imgs, marker_id, fast=False):
        self._check_img_input(imgs)

        poses = [
            c.module.estimate_marker_pose(img, marker_id) for c, img in zip(self.cameras, imgs)
        ]
        return self._filter_marker_poses(poses, fast=fast)

    # Visualization
    def visualize(self):
        viz = SceneViz()

        # Draw cameras
        for id in range(len(self.cameras)):
            viz.draw_camera(self.get_camera_pose(id))

        # Draw markers
        marker_dict = self.cameras[0].module.registered_markers
        for m_id in marker_dict:
            length = marker_dict[m_id]
            if length is not None:
                viz.draw_marker(self.get_marker_pose(m_id), m_id, length)

        # Show
        viz.show()

    # Save/Load
    def save_scene(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self.cameras, f)

    def load_scene(self, filename: str):
        with open(filename, "rb") as f:
            self.cameras = pickle.load(f)
        self.clear_snapshots()

    # Helper methods
    def _check_img_input(self, imgs):
        assert len(imgs) == self.get_num_cameras()

    def _combine_marker_detections(self, camera_outputs):
        marker_dict = {}
        for i, markers in enumerate(camera_outputs):
            for m in markers:
                if m.id not in marker_dict:
                    marker_dict[m.id] = m._asdict()
                    marker_dict[m.id].pop("pose")
                    marker_dict[m.id]["poses"] = [None for _ in camera_outputs]
                marker_dict[m.id]["poses"][i] = m.pose

        marker_ls = [marker_dict[key] for key in marker_dict]

        return marker_ls

    def _filter_marker_poses(self, poses, fast):
        # Check if at least one pose exists
        if len([pose for pose in poses if pose is not None]) == 0:
            return None

        # Filter
        if fast:  # Average estimations over local tangent space of one estimation
            trans_poses = []
            for i, pose in enumerate(poses):
                if pose is not None:
                    t_pose = self.cameras[i].pose * pose
                    trans_poses.append(t_pose)

            if len(trans_poses) == 1:
                return trans_poses[0]

            center_pose = trans_poses[0]
            delta_poses = [center_pose.inverse() * pose for pose in trans_poses]
            mean_delta_pose = sp.SE3.exp(np.mean([pose.log() for pose in delta_poses]))

            return center_pose * mean_delta_pose

        else:  # Full nonlinear least squares optimization
            graph = FactorGraph(len(self.cameras))
            for i, c in enumerate(self.cameras):
                graph.add_camera_prior(i, c.pose, definite=True)
            graph.add_marker(poses)
            result = graph.optimize()

            return result["markers"][0]


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
            self.ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color)

    def draw_axes(self, pose, length):
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

    def draw_camera(self, pose, color="grey", axes=True):
        # Draw a pyramid representing a camera
        b0_0 = CAMERA_SIZE * np.array([1, 1, 0])
        b1_0 = CAMERA_SIZE * np.array([1, -1, 0])
        b2_0 = CAMERA_SIZE * np.array([-1, -1, 0])
        b3_0 = CAMERA_SIZE * np.array([-1, 1, 0])
        t_0 = CAMERA_SIZE * np.array([0, 0, -2])

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
            self.draw_axes(pose, length=CAMERA_SIZE)

    def draw_marker(self, pose, id, length, color="k"):
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
        pos = pose.translation()
        self.ax.text(pos[0], pos[1], pos[2], id, color="b")

    def show(self):
        # Set limits
        mid = (self.max + self.min) / 2.0
        r = max(np.max(self.max - mid), np.max(mid - self.min))
        self.ax.set_xlim(mid[0] - r, mid[0] + r)
        self.ax.set_ylim(mid[1] - r, mid[1] + r)
        self.ax.set_zlim(mid[2] - r, mid[2] + r)

        # Show
        plt.show()
