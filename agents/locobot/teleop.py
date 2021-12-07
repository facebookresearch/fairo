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
from droidlet.dashboard.o3dviz import o3dviz
import numpy as np
from scipy.spatial import distance
import open3d as o3d

import time
import math

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()
    o3dviz.start()

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
            mover.bot.set_pan(mover.get_pan() + yaw)
            sync()
        elif command == "PAN_RIGHT":
            mover.bot.set_pan(mover.get_pan() - yaw)
            sync()
        elif command == "TILT_UP":
            mover.bot.set_tilt(mover.get_tilt() + yaw)
            print("action: TILT_UP", mover.get_tilt() + yaw)
            sync()
        elif command == "TILT_DOWN":
            mover.bot.set_tilt(mover.get_tilt() - yaw)
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
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 0.0.0.0",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--backend",
        help="Which backend to use: habitat (default), hellorobot",
        type=str,
        default='habitat',
    )
    args = parser.parse_args()
    
    ip = args.ip
    backend = args.backend
    
    print("Connecting to robot at ip: ", ip)

    if backend == 'habitat':
        from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
        mover = LoCoBotMover(ip=ip, backend='habitat')
    elif backend == 'hellorobot':
        from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover
        mover = HelloRobotMover(ip=ip)
    print("Mover is ready to be operated")

    log_settings = {
        "image_resolution": 512,  # pixels
        "image_quality": 10,  # from 10 to 100, 100 being best
    }

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
        
        o3dviz.put('pointcloud', opcd)

        # Plot the robot
        x, y, yaw = base_state.tolist()

        o3dviz.add_robot(base_state)
        o3dviz.add_axis()

        # start the SLAM
        if backend == 'habitat':
            mover.explore()
        
            # get the SLAM goals
            goal_loc, stg = None, None # mover.bot.get_slam_goal()    

            # plot the final goal
            if goal_loc is not None:
                goal_x, goal_y, goal_z = goal_loc
                cone = o3d.geometry.TriangleMesh.create_cylinder(radius=.2,
                                                                 height=3.,)
                cone.translate([goal_x, goal_y, 0.4], relative=False)
                cone.compute_vertex_normals()
                cone.paint_uniform_color([0.0, 1.0, 1.0])
                o3dviz.put('goal_cone', cone)

            # plot the short term goal in yellow and the path in green
            if stg is not None:
                stg_x, stg_y = stg
                cone = o3d.geometry.TriangleMesh.create_cylinder(radius=.2,
                                                                 height=3.,)
                cone.translate([stg_x, stg_y, 1.4], relative=False)
                cone.compute_vertex_normals()
                cone.paint_uniform_color([1.0, 1.0, 0.0])
                o3dviz.put('stg', cone)

                if prev_stg is None:
                    prev_stg = [y, -x]
                cur_stg = [stg_x, stg_y]

                arrow_length = distance.euclidean(cur_stg, prev_stg)
                if arrow_length > 0.0001:                
                    path = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=.03,
                                                                  cone_radius=.04,
                                                                  cylinder_height = arrow_length / 2,
                                                                  cone_height = arrow_length / 2,)
                    path.compute_vertex_normals()
                    path.paint_uniform_color([0.0, 1.0, 0.0])

                    path.translate([prev_stg[0], prev_stg[1], 0.2], relative=False)
                    path.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi/2, 0]))
                    path.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]))        
                    o3dviz.put('short_term_goal_path_{}'.format(path_count), path)
                    path_count = path_count + 1
                prev_stg = cur_stg

            # # get the obstacle map and plot it
            # obstacles = mover.bot.get_map()
            # obstacles = np.asarray(obstacles)
            # obstacles = np.concatenate((-obstacles[:, [1]], -obstacles[:, [0]], np.zeros((obstacles.shape[0], 1))), axis=1)
            # obspcd = o3d.geometry.PointCloud()
            # obspcd.points = o3d.utility.Vector3dVector(obstacles)
            # obspcd.paint_uniform_color([1.0, 0., 0.])
            # obsvox = o3d.geometry.VoxelGrid.create_from_point_cloud(obspcd, 0.03)
            # o3dviz.put('obstacles', obsvox)
        
        time.sleep(0.001)
