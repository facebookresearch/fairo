import copy
import pickle
from collections import namedtuple

import cv2
import numpy as np
import sophus as sp
import gtsam

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .camera import MarkerInfo
from .utils import sophus2gtsam, gtsam2sophus


DEFAULT_HUBER_C = 1.345
DEFAULT_POSE_SIG = [0.02, 0.02, 0.06, 0.1, 0.1, 0.1]
CAMERA_SIZE = 0.03


CameraInfo = namedtuple("CameraInfo", "module pose")
SnapshotInfo = namedtuple("SnapshotInfo", "id camera_outputs")


class Scene:
    def __init__(self, cameras=[]):
        self.cameras = [CameraInfo(c, sp.SE3()) for c in cameras]

        self.snapshots = {}
        self.snapshot_counter = 0

        self.origin_id = None

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

    # Scene calibration
    def calibrate_extrinsics(self, verbosity=0):
        assert len(self.snapshots) > 0, "At least 1 snapshot is required to calibrate extrinsics."

        graph = FactorGraph(len(self.cameras))

        # Parse all snapshots
        snapshot_markers = []
        for s_id in self.snapshots:
            markers = self._combine_marker_detections(self.snapshots[s_id].camera_outputs)
            snapshot_markers.append(markers)

        # Extract initial estimate of origin from snapshots
        pose_c = [
            sp.SE3() for _ in range(self.get_num_cameras())
        ]  # camera pose relative to origin marker

        if self.origin_id is None:  # Fixate first camera if no origin marker
            graph.add_camera_prior(0, sp.SE3(), definite=True)

        else:
            for m in snapshot_markers[0]:  # check first image for origin marker
                if m["id"] == self.origin_id:
                    for i, pose in enumerate(m["poses"]):
                        assert (
                            pose is not None
                        ), f"Failed to estimate pose of origin marker from camera {i} in snapshot."
                        pose_c[i] = pose.inverse()

        # Add all snapshots
        origin_idx = None
        for markers in snapshot_markers:
            for m in markers:
                if len([p for p in m["poses"] if p is not None]) == 0:
                    continue

                # Add marker to graph
                if m["id"] == self.origin_id:  # origin marker
                    if origin_idx is None:
                        origin_idx = graph.add_marker(m["poses"], init_guess=sp.SE3())
                        graph.add_marker_prior(origin_idx, sp.SE3(), definite=True)
                    else:
                        graph.add_marker(m["poses"], marker_idx=origin_idx)

                else:  # non-origin marker
                    init_guess = sp.SE3()
                    for i, pose in enumerate(m["poses"]):
                        if pose is not None:
                            init_guess = pose_c[i] * pose
                            break

                    graph.add_marker(m["poses"], init_guess=init_guess)

        # Optimize & record results
        results = graph.optimize(verbosity)
        for i, c in enumerate(self.cameras):
            self.cameras[i] = c._replace(pose=results["cameras"][i])

    # Marker registration
    def register_marker_size(self, marker_id, length):
        for c in self.cameras:
            c.module.register_marker_size(marker_id, length)

    def set_origin_marker(self, marker_id):
        self.origin_id = marker_id

    # Marker detection & estimation within scene
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
        self._visualize(cameras=self.cameras)

    def visualize_snapshot(self, imgs):
        self._check_img_input(imgs)
        markers = self.detect_markers(imgs)
        self._visualize(cameras=self.cameras, markers=markers)

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

    @staticmethod
    def _visualize(cameras=[], markers=[]):
        viz = SceneViz()

        # Draw cameras
        for c in cameras:
            viz.draw_camera(c.pose)

        # Draw markers
        for m in markers:
            if m.length:
                viz.draw_marker(m.pose, m.id, m.length)

        # Show
        viz.show()


class FactorGraph:
    def __init__(self, n_cameras):
        self.n_cameras = n_cameras

        # Setup variables & graph
        self.C = gtsam.symbol_shorthand.C  # camera
        self.M = gtsam.symbol_shorthand.M  # marker

        self.zero_pose_noise = gtsam.noiseModel.Constrained.All(6)
        self.pose_noise = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.Huber(DEFAULT_HUBER_C),
            gtsam.noiseModel.Diagonal.Sigmas(np.array(DEFAULT_POSE_SIG)),
        )

        self.graph = gtsam.NonlinearFactorGraph()

        # Initial estimate
        self.init_values = gtsam.Values()
        for i in range(self.n_cameras):
            self.init_values.insert(self.C(i), gtsam.Pose3())

        # Initialize
        self.n_samples = 0

    def add_camera_prior(self, camera_idx, pose, definite=True):
        assert camera_idx in range(self.n_cameras)
        gts_pose = sophus2gtsam(pose)

        # Add prior factor
        pose_noise = self.zero_pose_noise if definite else self.pose_noise
        factor = gtsam.PriorFactorPose3(self.C(camera_idx), gts_pose, pose_noise)
        self.graph.push_back(factor)

        # Update initial guess
        self.init_values.update(self.C(camera_idx), gts_pose)

    def add_marker_prior(self, marker_idx, pose, definite=False):
        assert marker_idx in range(self.n_samples)
        gts_pose = sophus2gtsam(pose)

        # Add prior factor
        pose_noise = self.zero_pose_noise if definite else self.pose_noise
        factor = gtsam.PriorFactorPose3(self.M(marker_idx), gts_pose, pose_noise)
        self.graph.push_back(factor)

        # Update initial guess
        self.init_values.update(self.M(marker_idx), gts_pose)

    def add_marker(self, pose_ls, init_guess=sp.SE3(), marker_idx=None):
        assert len(pose_ls) == self.n_cameras

        # Idx not specified => new data point
        if marker_idx is None:
            idx = self.n_samples
            self.n_samples += 1
        else:
            assert marker_idx in range(self.n_samples)
            idx = marker_idx

        # Create between factor
        for i in range(self.n_cameras):
            if pose_ls[i] is None:
                continue
            factor = gtsam.BetweenFactorPose3(
                self.C(i), self.M(idx), sophus2gtsam(pose_ls[i]), self.pose_noise
            )
            self.graph.push_back(factor)

        # Initial estimate
        if marker_idx is None:
            self.init_values.insert(self.M(idx), sophus2gtsam(init_guess))
        else:
            self.init_values.update(self.M(idx), sophus2gtsam(init_guess))

        return idx

    def optimize(self, verbosity=0):
        # Setup optimization
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity(["SILENT", "TERMINATION"][verbosity])
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.init_values, params)

        # Optimize
        if verbosity > 0:
            print("Optimizing extrinsics...")
        result = optimizer.optimize()
        if verbosity > 0:
            print(f"initial error = {self.graph.error(self.init_values)}")
            print(f"final error = {self.graph.error(result)}")

        # Format result
        camera_poses = [gtsam2sophus(result.atPose3(self.C(i))) for i in range(self.n_cameras)]
        marker_poses = [gtsam2sophus(result.atPose3(self.M(j))) for j in range(self.n_samples)]

        return {
            "cameras": camera_poses,
            "markers": marker_poses,
        }


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
