from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

import numpy as np
import sophus as sp

from .camera import MarkerInfo
from .graph import FactorGraph
from .viz import SceneViz


DEFAULT_MARKER_LENGTH = 0.05
DEFAULT_CAMERA_SIZE = 0.08

DEFAULT_CAMERA_NOISE = [0.01, 0.01, 0.05, 0.1, 0.1, 0.1]  # more uncertainty in z direction
DEFAULT_CALIB_NOISE = [0.002, 0.002, 0.002, 0.02, 0.02, 0.02]


class ObjectType(Enum):
    CAMERA = 1
    MARKER = 2


@dataclass
class Object:
    name: str
    obj_type: ObjectType
    frame: str
    pose: sp.SE3
    pose_in_frame: sp.SE3
    size: float
    is_anchor: bool = False
    is_visible: bool = False


@dataclass
class Frame:
    name: str
    pose: sp.SE3
    objects: List[str] = field(default_factory=list)
    is_visible: bool = False


class Scene:
    def __init__(self, camera_noise=None, calib_noise=None):
        # Initialize data containers
        self._frames = {}
        self._objects = {}

        # Noise
        if camera_noise is None:
            self._camera_noise = np.array(DEFAULT_CAMERA_NOISE)
        else:
            assert len(camera_noise) == 6, "Invalid noise vector dimensions."
            self._camera_noise = np.array(camera_noise)

        if calib_noise is None:
            self._calib_noise = np.array(DEFAULT_CALIB_NOISE)
        else:
            assert len(calib_noise) == 6, "Invalid noise vector dimensions."
            self._calib_noise = np.array(calib_noise)

        # Default world frame
        f0 = Frame("world", sp.SE3())
        self._frames["world"] = f0

    # Scene construction
    def _add_object(self, name, obj_type, frame, pose_in_frame, size):
        # Parse input
        assert name not in self._objects.keys(), f"Object name already exists: {name}"
        assert frame in self._frames.keys(), f"Unknown frame: {frame}"

        is_anchor = pose_in_frame is not None
        pose_in_frame = sp.SE3() if pose_in_frame is None else pose_in_frame
        pose = self._frames[frame].pose * pose_in_frame

        # Add to data
        obj = Object(
            name,
            obj_type,
            frame,
            pose,
            pose_in_frame=pose_in_frame,
            size=size,
            is_anchor=is_anchor,
        )
        self._objects[name] = obj
        self._frames[frame].objects.append(name)

    def add_camera(self, name: str, frame="world", pose_in_frame=None, size=DEFAULT_CAMERA_SIZE):
        self._add_object(name, ObjectType.CAMERA, frame, pose_in_frame, size)

    def add_marker(self, id: int, frame="world", pose_in_frame=None, length=DEFAULT_MARKER_LENGTH):
        self._add_object(
            str(id),
            obj_type=ObjectType.MARKER,
            frame=frame,
            pose_in_frame=pose_in_frame,
            size=length,
        )

    def add_frame(self, name, pose=None):
        pose = sp.SE3() if pose is None else pose

        f = Frame(name, pose)
        self._frames[name] = f

    # Get scene info
    def get_markers(self):
        return [
            int(name) for name, obj in self._objects.items() if obj.obj_type == ObjectType.MARKER
        ]

    def get_cameras(self):
        return [name for name, obj in self._objects.items() if obj.obj_type == ObjectType.CAMERA]

    def get_frames(self):
        return list(self._frames.keys())

    def get_marker_info(self, id: int):
        name = str(id)
        assert name in self._objects.keys(), f"Unknown marker: {id}"
        marker = self._objects[name]
        return {
            "id": int(marker.name),
            "frame": marker.frame,
            "pose": marker.pose,
            "pose_in_frame": marker.pose_in_frame,
            "length": marker.size,
            "is_anchor": marker.is_anchor,
            "is_visible": marker.is_visible,
        }

    def get_camera_info(self, name: str):
        assert name in self._objects.keys(), f"Unknown camera: {name}"
        camera = self._objects[name]
        return {
            "name": camera.name,
            "frame": camera.frame,
            "pose": camera.pose,
            "pose_in_frame": camera.pose_in_frame,
            "is_anchor": camera.is_anchor,
        }

    def get_frame_info(self, name: str):
        assert name in self._frames.keys(), f"Unknown frame: {name}"
        frame = self._frames[name]

        markers = []
        cameras = []
        for obj_name in frame.objects:
            obj = self._objects[obj_name]
            if obj.obj_type == ObjectType.MARKER:
                markers.append(int(obj.name))
            else:
                cameras.append(obj.name)

        return {
            "name": frame.name,
            "pose": frame.pose,
            "markers": markers,
            "cameras": cameras,
        }

    def get_frame_poses(self):
        return {frame_name: frame.pose for frame_name, frame in self._frames.items()}

    # Graph operations
    def _reset_visibility(self):
        for frame_name, frame in self._frames.items():
            frame.is_visible = False
        for obj_name, obj in self._objects.items():
            obj.is_visible = False

    def _init_frame(self, graph, frame, prefix, lock_frames=True, cost_multiplier=1.0):
        f = self._frames[frame]
        f.is_visible = True
        f_node = f"f_{prefix}_{frame}"
        graph.init_variable(f_node, f.pose)

        for object_name in f.objects:
            o = self._objects[object_name]
            o.is_visible = True
            o_node = f"o_{prefix}_{o.name}"

            graph.init_variable(o_node, o.pose)
            if lock_frames or o.is_anchor:
                graph.add_observation(
                    f_node, o_node, o.pose_in_frame, self._calib_noise / cost_multiplier
                )
            else:
                t_node = f"t_{o.name}"
                graph.init_variable(t_node, o.pose_in_frame)
                graph.add_fixed_transform(
                    f_node, o_node, t_node, self._calib_noise / cost_multiplier
                )

    def _add_world_prior(self, graph, lock_frames, cost_multiplier=1.0):
        self._init_frame(
            graph, "world", prefix="", lock_frames=lock_frames, cost_multiplier=cost_multiplier
        )
        graph.add_prior("f__world", sp.SE3())

    def _add_detected_markers(self, graph, detected_markers, prefix, lock_frames):
        updated_frames = set("world")

        for camera_name, markers in detected_markers.items():
            # Init camera
            c = self._objects[camera_name]
            if c.frame == "world":
                c_node = f"o__{camera_name}"
            else:
                c_node = f"o_{prefix}_{camera_name}"
            if c.frame not in updated_frames:
                self._init_frame(graph, c.frame, prefix=prefix, lock_frames=lock_frames)
                updated_frames.add(c.frame)

            for marker_obs in markers:
                marker_name = str(marker_obs.id)
                if marker_obs.pose is None or str(marker_obs.id) not in self._objects:
                    continue

                # Init marker
                m = self._objects[marker_name]
                if m.frame == "world":
                    m_node = f"o__{marker_name}"
                else:
                    m_node = f"o_{prefix}_{marker_name}"
                if m.frame not in updated_frames:
                    self._init_frame(graph, m.frame, prefix=prefix, lock_frames=lock_frames)
                    updated_frames.add(m.frame)

                # Add observations
                graph.add_observation(c_node, m_node, marker_obs.pose, self._camera_noise)

    def _optimize_and_update(self, graph, verbosity=0):
        # Optimize graph
        results = graph.optimize(verbosity=verbosity)

        # Extract results
        for frame_name, frame in self._frames.items():
            f_node = f"f__{frame_name}"
            if f_node in results:
                frame.pose = results[f_node]

        for obj_name, obj in self._objects.items():
            t_node = f"t_{obj_name}"
            if t_node in results:
                obj.pose_in_frame = results[t_node]

                frame = self._frames[obj.frame]
                obj.pose = frame.pose * obj.pose_in_frame

            o_node = f"o__{obj_name}"
            if o_node in results:
                obj.pose = results[o_node]

    def update_pose_estimations(self, detected_markers: Dict[str, List[MarkerInfo]]):
        """ Estimate relative poses between frames """
        graph = FactorGraph()

        # Reset visibility
        self._reset_visibility()

        # Add factors
        self._add_world_prior(graph, lock_frames=True)
        self._add_detected_markers(graph, detected_markers, prefix="", lock_frames=True)

        # Optimize graph & update data
        self._optimize_and_update(graph)

    def calibrate_extrinsics(
        self, detected_markers_ls: List[Dict[str, List[MarkerInfo]]], verbosity=0
    ):  # TODO: Add aux info for frames
        """ Calibrate extrinsics between cameras & markers in each frame """
        graph = FactorGraph()

        # Reset visibility
        self._reset_visibility()

        # Add factors
        n_samples = len(detected_markers_ls)
        self._add_world_prior(graph, lock_frames=False, cost_multiplier=n_samples)

        for i, detected_markers in enumerate(detected_markers_ls):
            self._add_detected_markers(graph, detected_markers, prefix=i, lock_frames=False)

        # Initialize variables using BFS
        prior_node = "f__world"
        queue = [(prior_node, sp.SE3())]
        visited = set([prior_node])

        while queue:
            curr_node, pose = queue.pop(0)
            graph.init_variable(curr_node, pose)

            for next_node, transform in graph.factor_edges[curr_node]:
                if next_node not in visited:
                    queue.append((next_node, pose * transform))
                    visited.add(curr_node)

        # Optimize graph & update data
        self._optimize_and_update(graph, verbosity)

    # Rendering
    def render_scene(self, show_marker_id=False):
        viz = SceneViz()

        # Draw markers & cameras
        for obj_name, obj in self._objects.items():
            if not obj.is_visible:
                continue
            if obj.obj_type == ObjectType.MARKER:
                viz.draw_marker(
                    obj.pose, id=int(obj.name), length=obj.size, show_id=show_marker_id
                )
            else:
                viz.draw_camera(obj.pose, size=obj.size)

        # Draw frames
        for frame_name, frame in self._frames.items():
            if not frame.is_visible:
                continue
            viz.draw_axes(frame.pose)

        # Show
        viz.show()

    # Save / Load TODO
    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    def save_scene(self, filename):
        pass

    def load_scene(self, filename):
        pass
