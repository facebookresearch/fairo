import os
import subprocess
import time
import signal
import random
import logging
import faulthandler
import threading
import functools

from multiprocessing import set_start_method

from droidlet import dashboard
import numpy as np
from scipy.spatial import distance

os.environ["WEBRTC_IP"] = "0.0.0.0"
os.environ["WEBRTC_PORT"] = "8889"

import open3d as o3d
o3d.visualization.webrtc_server.enable_webrtc()

from open3d.visualization import O3DVisualizer, gui
import threading
import queue
import time
import math

class O3dViz(threading.Thread):
    def __init__(self, *args, **kwargs):
        self.q = queue.Queue()
        super().__init__(*args, **kwargs)

    def put(self, name, command, obj):
        # pass
        self.q.put([name, command, obj])

    def run(self):
        app = gui.Application.instance

        app.initialize()
        w = O3DVisualizer("o3dviz", 1024, 768)
        w.set_background((0.0, 0.0, 0.0, 1.0), None)

        app.add_window(w)
        reset_camera = False

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

                except:
                    print("failed to add geometry to scene")
                if not reset_camera:
                    # Look at A from camera placed at B with Y axis
                    # pointing at C
                    # useful for pyrobot co-ordinates
                    w.scene.camera.look_at([1, 0, 0],
                                           [-5, 0, 1],
                                           [0, 0, 1])
                    # useful for initial camera co-ordinates
                    # w.scene.camera.look_at([0, 0, 1],
                    #                        [0, 0, -1],
                    #                        [0, -1, 0])
                    reset_camera = True
                w.post_redraw()
            except queue.Empty:
                pass


if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()

from droidlet.interpreter.robot import (
    dance, 
    default_behaviors,
    LocoGetMemoryHandler, 
    PutMemoryHandler, 
    LocoInterpreter,
)
from droidlet.dialog.robot import LocoBotCapabilities
from droidlet.event import sio

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()


mover = None

@sio.on("sendCommandToAgent")
def get_command(sid, command):
    command, value = command.split()
    print(command)
    print(value)
    test_command(sid, [command], value=value)

@sio.on("logData")
def log_data(sid, seconds):
    test_command(sid, ["LOG_DATA"], value=seconds)

@sio.on("stopRobot")
def stop_robot(sid):
    test_command(sid, ["STOP_ROBOT"])

@sio.on("unstopRobot")
def unstop_robot(sid):
    test_command(sid, ["UNSTOP_ROBOT"])


def test_command(sid, commands, data={"yaw": 0.1, "velocity": 0.1, "move": 0.3}, value=None):
    print(commands, data, value)
    move_dist = float(data['move'])
    yaw = float(data['yaw'])
    velocity = float(data['velocity'])
    global mover
    if mover == None:
        return
    if value is not None:
        move_dist = value

    def sync():
        time.sleep(10)
        for i in range(50):
            mover.get_rgb_depth()

    movement = [0.0, 0.0, 0.0]
    for command in commands:
        if command == "MOVE_FORWARD":
            movement[0] += float(move_dist)
            print("action: FORWARD", movement)
            mover.move_relative([movement], blocking=False)
        elif command == "MOVE_BACKWARD":
            movement[0] -= float(move_dist)
            print("action: BACKWARD", movement)
            mover.move_relative([movement], blocking=False)
        elif command == "MOVE_LEFT":
            movement[2] += yaw
            print("action: LEFT", movement)
            mover.move_relative([movement], blocking=False)
        elif command == "MOVE_RIGHT":
            movement[2] -= yaw
            print("action: RIGHT", movement)
            mover.move_relative([movement], blocking=False)
        elif command == "PAN_LEFT":
            mover.bot.set_pan(mover.bot.get_pan().value + yaw).value
            sync()
        elif command == "PAN_RIGHT":
            mover.bot.set_pan(mover.bot.get_pan().value - yaw).value
            sync()
        elif command == "TILT_UP":
            mover.bot.set_tilt(mover.bot.get_tilt().value + yaw).value
            sync()
        elif command == "TILT_DOWN":
            mover.bot.set_tilt(mover.bot.get_tilt().value - yaw).value
            sync()
        elif command == "LOG_DATA":
            mover.log_data_start(float(value)) # in seconds
        elif command == "STOP_ROBOT":
            mover.stop()
        elif command == "UNSTOP_ROBOT":
            mover.unstop()
        elif command == "SET_PAN":
            print("action: SET_PAN", float(value))
            mover.bot.set_pan(float(value))
            sync()
        elif command == "SET_TILT":
            print("action: SET_TILT", float(value))
            mover.bot.set_tilt(float(value))
            sync()
        elif command == "MOVE_ABSOLUTE":
            xyyaw_s = value.split(',')
            xyyaw_f = [float(v) for v in xyyaw_s]
            print("action: MOVE_ABSOLUTE", xyyaw_f)
            mover.move_absolute(xyyaw_f, blocking=False)
            sync()
        elif command == "LOOK_AT":
            xyz = value.split(',')
            xyz = [float(p) for p in xyz]
            print("action: LOOK_AT", xyz)
            mover.look_at(xyz, turn_base=False)
        elif command == "RESET":
            mover.bot.set_tilt(0.)
            mover.bot.set_pan(0.)

        print(command, movement)

@sio.on("movement command")
def test_command_web(sid, commands, data, value=None):
    test_command(sid, commands, data=data, value=value)


if __name__ == "__main__":
    ip = os.getenv("LOCOBOT_IP")
    print("Connecting to robot at ip: ", ip)
    from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover
    mover = HelloRobotMover(ip=ip)
    print("Mover is ready to be operated")

    log_settings = {
        "image_resolution": 512,  # pixels
        "image_quality": 10,  # from 10 to 100, 100 being best
    }

    o3dviz = O3dViz()
    o3dviz.start()

    all_points = None
    all_colors = None
    first = True
    prev_stg = None
    path_count = 0

    start_time = time.time_ns()
    fps_freq = 1 # displays the frame rate every 1 second
    counter = 0
    
    while True:
        counter += 1
        iter_time = time.time_ns() - start_time
        if float(iter_time) / 1e9 > fps_freq :
            print("FPS: ", round(counter / (float(iter_time) / 1e9), 1), "  ", int(iter_time / 1e6 / counter), "ms")
            counter = 0
            start_time = time.time_ns()

        base_state = mover.get_base_pos_in_canonical_coords()

        sio.emit("image_settings", log_settings)
        resolution = log_settings["image_resolution"]
        quality = log_settings["image_quality"]

        # this goes from 21ms to 120ms
        rgb_depth = mover.get_rgb_depth()

        # this takes about 1.5 to 2 fps
        serialized_image = rgb_depth.to_struct(resolution, quality)

        sio.emit("rgb", serialized_image["rgb"])
        sio.emit("depth", {
            "depthImg": serialized_image["depth_img"],
            "depthMax": serialized_image["depth_max"],
            "depthMin": serialized_image["depth_min"],
        })


        points, colors = rgb_depth.ptcloud.reshape(-1, 3), rgb_depth.rgb.reshape(-1, 3)
        colors = colors / 255.

        if all_points is None:
            all_points = points
            all_colors = colors
        else:
            all_points = np.concatenate((all_points, points), axis=0)
            all_colors = np.concatenate((all_colors, colors), axis=0)

        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(all_points)
        opcd.colors = o3d.utility.Vector3dVector(all_colors)
        opcd = opcd.voxel_down_sample(0.05)

        # # remove the rooftop / ceiling points in the point-cloud to make it easier to see the robot in the visualization
        # crop_bounds = o3d.utility.Vector3dVector([
        #     [-1000., -20., -1000.],
        #     [1000., 20., 1000.0],
        #     ])
        # opcd = opcd.crop(
        #     o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        #         crop_bounds,
        #     )
        # )
        
        
        all_points = np.asarray(opcd.points)
        all_colors = np.asarray(opcd.colors)
        
        if first:
            cmd = 'add'
            first = False
        else:
            cmd = 'replace'
            
        o3dviz.put('pointcloud', cmd, opcd)

        # Plot the robot
        x, y, yaw = base_state.tolist()

        robot_orientation = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=.05,
                                                       cone_radius=.075,
                                                       cylinder_height = .50,
                                                       cone_height = .4,
                                                       resolution=20)
        robot_orientation.compute_vertex_normals()
        robot_orientation.paint_uniform_color([1.0, 0.5, 0.1])
        
        robot_orientation.translate([y, -x, 0.], relative=False)
        robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi/2, 0]))
        if yaw != 0:
            robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]))        

        o3dviz.put('bot_orientation', cmd, robot_orientation)

        robot_base = o3d.geometry.TriangleMesh.create_cylinder(radius=.1,
                                                          height=1,)
        robot_base.translate([y, -x, 0.1], relative=False)
        robot_base.compute_vertex_normals()
        robot_base.paint_uniform_color([1.0, 1.0, 0.1])

        o3dviz.put('bot_base', cmd, robot_base)



        # red = x, green = y, blue = z
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))        
        axis.compute_vertex_normals()
        o3dviz.put('axis', cmd, axis)

        
        time.sleep(0.001)
