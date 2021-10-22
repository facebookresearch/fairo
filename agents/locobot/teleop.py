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
                    w.reset_camera_to_default()
                    reset_camera = True
                w.post_redraw()
            except queue.Empty:
                pass


if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()

# from droidlet.dialog.dialogue_manager import DialogueManager
# from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
# from droidlet.base_util import to_player_struct, Pos, Look, Player
# from droidlet.memory.memory_nodes import PlayerNode
# from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
# from droidlet.memory.robot.loco_memory import LocoAgentMemory, DetectedObjectNode
# from droidlet.perception.robot import Perception
# from self_perception import SelfPerception
from droidlet.interpreter.robot import (
    dance, 
    default_behaviors,
    LocoGetMemoryHandler, 
    PutMemoryHandler, 
    LocoInterpreter,
)
from droidlet.dialog.robot import LocoBotCapabilities
# import droidlet.lowlevel.rotation as rotation
from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
# from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover
from droidlet.event import sio

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()


mover = None

@sio.on("movement command")
def test_command(sid, commands, yaw_velocity):
    print(commands, yaw_velocity)
    global mover
    if mover == None:
        return
    movement = [0.0, 0.0, 0.0]
    for command in commands:
        if command == "MOVE_FORWARD":
            movement[0] += 0.1
            print("action: FORWARD")
        elif command == "MOVE_BACKWARD":
            movement[0] -= 0.1
            print("action: BACKWARD")
        elif command == "MOVE_LEFT":
            movement[2] += 0.3
            print("action: LEFT")
        elif command == "MOVE_RIGHT":
            movement[2] -= 0.3
            print("action: RIGHT")
        elif command == "PAN_LEFT":
            mover.bot.set_pan(mover.bot.get_pan() + 0.08)
        elif command == "PAN_RIGHT":
            mover.bot.set_pan(mover.bot.get_pan() - 0.08)
        elif command == "TILT_UP":
            mover.bot.set_tilt(mover.bot.get_tilt() - 0.08)
        elif command == "TILT_DOWN":
            mover.bot.set_tilt(mover.bot.get_tilt() + 0.08)
        print(command, movement)
        mover.move_relative([movement])


if __name__ == "__main__":
    ip = os.getenv("LOCOBOT_IP")
    print("Connecting to robot at ip: ", ip)
    # mover = HelloRobotMover(ip=ip)
    mover = LoCoBotMover(ip=ip)
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

    start_time = time.time()
    fps_freq = 1 # displays the frame rate every 1 second
    counter = 0
    
    while True:
        counter += 1
        if (time.time() - start_time) > fps_freq :
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()

        movement = [0.0, 0.0, 0.3]
        # mover.move_relative([movement], use_dslam=False)

        base_state = mover.get_base_pos_in_canonical_coords()
        # print("base_state: ", base_state)
        # print("rgb_depth: ", mover.get_rgb_depth())
        # sio.emit("image_settings", log_settings)
        # resolution = log_settings["image_resolution"]
        # quality = log_settings["image_quality"]

        # rgb_depth = mover.get_rgb_depth()
        # serialized_image = rgb_depth.to_struct(resolution, quality)

        # sio.emit("rgb", serialized_image["rgb"])
        # sio.emit("depth", {
        #     "depthImg": serialized_image["depth_img"],
        #     "depthMax": serialized_image["depth_max"],
        #     "depthMin": serialized_image["depth_min"],
        # })

        pcd = mover.get_current_pcd(in_global=True)
        points = pcd[0]
        colors = pcd[1]/255.

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


        # remove the rooftop / ceiling points in the point-cloud to make it easier to see the robot in the visualization
        crop_bounds = o3d.utility.Vector3dVector([
            [-1000., -1000., -1000.],
            [1000., 1000., 2.0],
            ])
        opcd = opcd.crop(
            o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                crop_bounds,
            )
        )

        all_points = np.asarray(opcd.points)
        all_colors = np.asarray(opcd.colors)

        # print(all_points.shape)


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
        robot_orientation.paint_uniform_color([0.1, 0.9, 0.1])
        
        robot_orientation.translate([y, -x, 0.5], relative=False)
        robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi/2, 0]))
        robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]))        

        o3dviz.put('bot_orientation', cmd, robot_orientation)

        robot_base = o3d.geometry.TriangleMesh.create_cylinder(radius=.1,
                                                          height=1.,)
        robot_base.translate([y, -x, 0.4], relative=False)
        robot_base.compute_vertex_normals()
        robot_base.paint_uniform_color([1.0, 1.0, 0.1])

        o3dviz.put('bot_base', cmd, robot_base)


        # start the SLAM
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
            o3dviz.put('goal_cone', cmd, cone)

        # plot the short term goal in yellow and the path in green
        if stg is not None:
            stg_x, stg_y = stg
            cone = o3d.geometry.TriangleMesh.create_cylinder(radius=.2,
                                                             height=3.,)
            cone.translate([stg_x, stg_y, 1.4], relative=False)
            cone.compute_vertex_normals()
            cone.paint_uniform_color([1.0, 1.0, 0.0])
            o3dviz.put('stg', cmd, cone)

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
                o3dviz.put('short_term_goal_path_{}'.format(path_count), 'add', path)
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
        # o3dviz.put('obstacles', cmd, obsvox)                
            
        
        time.sleep(0.001)
