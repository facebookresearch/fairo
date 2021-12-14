import os
import time

os.environ["WEBRTC_IP"] = "0.0.0.0"
os.environ["WEBRTC_PORT"] = "8889"

import open3d as o3d
o3d.visualization.webrtc_server.enable_webrtc()

from open3d.visualization import O3DVisualizer, gui
import threading
import queue
import math


class O3dViz(threading.Thread):
    def __init__(self, *args, **kwargs):
        self.q = queue.Queue()
        super().__init__(*args, **kwargs)
        self.look_at = [1, 0, 0]  # look at x positive
        self.cam_pos = [-5, 0, 1] # 5 cm behind origin
        self.y_axis = [0, 0, 1]   # y axis is z-inward
        self.keys = set()

    def put(self, name, obj):
        cmd = 'add'
        if name in self.keys:
            cmd = 'replace'
        else:
            self.keys.add(name)
        self.q.put([name, cmd, obj])

    def set_camera(self, look_at, position, y_axis):
        self.look_at = look_at
        self.cam_pos = position
        self.y_axis = y_axis
        self.reset_camera = True

    def add_robot(self, base_state, base=True, canonical=True):
        x, y, yaw = base_state.tolist()
        if canonical:
            x_old, y_old = x, y
            x, y = y, -x

        robot_orientation = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=.05,
                                                                   cone_radius=.075,
                                                                   cylinder_height = .50,
                                                                   cone_height = .4,
                                                                   resolution=20)
        robot_orientation.compute_vertex_normals()
        robot_orientation.paint_uniform_color([1.0, 0.5, 0.1])

        robot_orientation.translate([x, y, 0.], relative=False)
        # make the cylinder representing the robot to be parallel to the floor
        robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi/2, 0]))
        # rotate the cylinder by the robot orientation
        if yaw != 0:
            robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]))

        self.put('bot_orientation', robot_orientation)

        if base:
            height = 1.41 # hello-robot stretch is 141 cms high
            radius = 0.34 / 2 # hello-robot stretch is 34cm x 33 cm footprint
            robot_base = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                       height=height,)
            robot_base.translate([x, y, height / 2.0], relative=False)
            robot_base.compute_vertex_normals()
            robot_base.paint_uniform_color([1.0, 1.0, 0.1])
            self.put('bot_base', robot_base)

    def add_axis(self):
        # red = x, green = y, blue = z
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
        axis.compute_vertex_normals()
        self.put('axis', axis)

    def run(self):        
        app = gui.Application.instance

        app.initialize()
        w = O3DVisualizer("o3dviz", 1024, 768)
        w.set_background((1.0, 1.0, 1.0, 1.0), None)
        w.ground_plane = o3d.visualization.rendering.Scene.GroundPlane(1) # XY is the ground plane
        w.show_ground = True        
        w.show_axes = True
        w.mouse_mode = o3d.visualization.gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE
         
        app.add_window(w)
        self.reset_camera = False
        self.first_object_added = False

        while True:
            app.run_one_tick()            
            time.sleep(0.001)

            try:
                name, command, geometry = self.q.get_nowait()

                try:
                    if command == 'remove':
                        w.remove_geometry(name)
                    elif command == 'replace':
                        w.remove_geometry(name)
                        w.add_geometry(name, geometry)
                    elif command == 'add':
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
                # w.scene.camera.set_projection(120, 4./3., 0.0001, 1000000000000000., o3d.visualization.rendering.Camera.FovType.Horizontal)                    
                w.post_redraw()
            except queue.Empty:
                pass


o3dviz = O3dViz()

def start():
    o3dviz.start()
