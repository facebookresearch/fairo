import os
import time
import pickle
import queue
import math
import functools

import numpy as np
import torch
import open3d as o3d
from open3d.visualization import O3DVisualizer, gui
from droidlet.parallel import BackgroundTask

attributes = {
    "TriangleMesh": {
        "vertices": o3d.utility.Vector3dVector,
        "triangles": o3d.utility.Vector3iVector,
        "vertex_colors": o3d.utility.Vector3dVector,
        "vertex_normals": o3d.utility.Vector3dVector,
    },
    "PointCloud": {
        "points": o3d.utility.Vector3dVector,
        "colors": o3d.utility.Vector3dVector,
    },
    "OrientedBoundingBox": {
        "center": np.ndarray,
        "color": np.ndarray,
        "extent": np.ndarray,
        "R": np.ndarray,
    },
}


def serialize(m):
    class_type = type(m)
    class_name = class_type.__name__
    class_attrs = attributes[class_name]
    d = {}
    for name, typ in class_attrs.items():
        val = getattr(m, name)
        if not typ == type(val):
            raise RuntimeError(
                "for {}.{}, expected {}, but got {}".format(class_name, name, typ, type(val))
            )
        d[name] = torch.from_numpy(np.asarray(val))
        d[name].share_memory_()
    ser = pickle.dumps([class_name, d])
    return ser


def deserialize(obj):
    class_name, ser = pickle.loads(obj)
    class_attrs = attributes[class_name]
    m = getattr(o3d.geometry, class_name)()
    for name, typ in class_attrs.items():
        attr = ser[name]
        attr = attr.numpy()
        if typ == np.ndarray:
            typed_attr = attr
        else:
            typed_attr = typ(attr)
        setattr(m, name, typed_attr)
    return m


class O3dViz:
    def __init__(self, *args, **kwargs):
        self.q = queue.Queue()
        super().__init__(*args, **kwargs)
        self.look_at = [1, 0, 0]  # look at x positive
        self.cam_pos = [-5, 0, 1]  # 5 cm behind origin
        self.y_axis = [0, 0, 1]  # y axis is z-inward
        self.keys = set()
        self._init = False
        self.counter = 0

    def put(self, name, obj):
        cmd = "add"
        if name in self.keys:
            cmd = "replace"
        else:
            self.keys.add(name)
        self.q.put([name, cmd, obj])

    def remove(self, name):
        cmd = "remove"
        self.keys.discard(name)
        self.q.put([name, cmd, None])

    def set_camera(self, look_at, position, y_axis):
        self.look_at = look_at
        self.cam_pos = position
        self.y_axis = y_axis
        self.reset_camera = True

    def add_robot(self, base_state, base=True, canonical=True, height=1.41):
        x, y, yaw = base_state.tolist()
        if canonical:
            x_old, y_old = x, y
            x, y = y, -x

        robot_orientation = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.05,
            cone_radius=0.075,
            cylinder_height=0.50,
            cone_height=0.4,
            resolution=20,
        )
        robot_orientation.compute_vertex_normals()
        robot_orientation.paint_uniform_color([1.0, 0.5, 0.1])

        robot_orientation.translate([x, y, 0.0], relative=False)
        # make the cylinder representing the robot to be parallel to the floor
        robot_orientation.rotate(
            o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi / 2, 0])
        )
        # rotate the cylinder by the robot orientation
        if yaw != 0:
            robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]))

        self.put("bot_orientation", robot_orientation)

        if base:
            height = height  # hello-robot stretch is 141 cms high
            radius = 0.34 / 2  # hello-robot stretch is 34cm x 33 cm footprint
            robot_base = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius,
                height=height,
            )
            robot_base.translate([x, y, height / 2.0], relative=False)
            robot_base.compute_vertex_normals()
            robot_base.paint_uniform_color([1.0, 1.0, 0.1])
            self.put("bot_base", robot_base)

    def init(self):
        if self._init:
            return
        app = gui.Application.instance
        self.app = app

        app.initialize()
        w = O3DVisualizer("o3dviz", 800, 600)
        self.w = w
        w.set_background((1.0, 1.0, 1.0, 1.0), None)
        w.ground_plane = o3d.visualization.rendering.Scene.GroundPlane(1)  # XY is the ground plane
        w.show_ground = True
        w.show_axes = True
        w.mouse_mode = o3d.visualization.gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE

        app.add_window(w)
        self.reset_camera = False
        self.first_object_added = False
        while not self.q.empty():
            self.q.get()
        self._init = True
        self.start_time = time.time_ns()

    def run(self):
        while True:
            self.run_tick(threaded=True)

    def run_tick(self, threaded=False):
        self.init()

        app, w = self.app, self.w

        app.run_one_tick()

        if threaded:
            time.sleep(0.001)

        self.counter += 1
        iter_time = time.time_ns() - self.start_time
        if float(iter_time) / 1e9 > 1:
            # print("Drawing FPS: ", round(self.counter / (float(iter_time) / 1e9), 1), "  ", int(iter_time / 1e6 / self.counter), "ms")
            self.counter = 0
            self.start_time = time.time_ns()

        try:
            name, command, geometry = self.q.get_nowait()

            try:
                if command == "remove":
                    w.remove_geometry(name)
                elif command == "replace":
                    w.remove_geometry(name)
                    w.add_geometry(name, geometry)
                elif command == "add":
                    w.add_geometry(name, geometry)
                    if not self.first_object_added:
                        w.reset_camera_to_default()
                        self.first_object_added = True

            except:
                print("failed to add geometry to scene")
            if self.reset_camera:
                # Look at A from camera placed at B with Y axis
                # pointing at C
                # useful for pyrobot co-ordinates
                w.scene.camera.look_at(self.look_at, self.cam_pos, self.y_axis)

                # useful for initial camera co-ordinates
                # w.scene.camera.look_at([0, 0, 1],
                #                        [0, 0, -1],
                #                        [0, -1, 0])
                self.reset_camera = False
        except queue.Empty:
            pass
        w.post_redraw()


def viz_init(webrtc_streaming=True):
    if webrtc_streaming:
        os.environ["WEBRTC_IP"] = "0.0.0.0"
        os.environ["WEBRTC_PORT"] = "8889"
        o3d.visualization.webrtc_server.enable_webrtc()
    o3dviz = O3dViz()
    o3dviz.init()
    return o3dviz


def viz_tick(o3dviz, command=None):
    if command is not None:
        name, ser = command
        if name == "add_robot":
            o3dviz.add_robot(*ser)
        elif name == "remove":
            o3dviz.remove(ser)
        else:
            geometry = deserialize(ser)
            o3dviz.put(name, geometry)
    o3dviz.run_tick(threaded=False)


class O3DVizProcess(BackgroundTask):
    def put(self, name, geometry):
        try:
            super().get_nowait()
        except queue.Empty:
            pass

        ser = serialize(geometry)
        super().put([name, ser])

    def remove(self, name):
        super().put(["remove", name])

    def add_robot(self, base_state, base=True, canonical=True, height=1.41):
        super().put(["add_robot", [base_state, base, canonical, height]])

    def start(self):
        super().start(exec_empty=True)


def O3DViz(webrtc_streaming=True):
    init_args = [webrtc_streaming]
    init_fn = viz_init
    process_fn = viz_tick
    return O3DVizProcess(init_fn=viz_init, init_args=init_args, process_fn=viz_tick)
