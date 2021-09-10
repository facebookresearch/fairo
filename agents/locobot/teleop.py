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
                if command == 'remove':
                    w.remove_geometry(name)
                elif command == 'replace':
                    w.remove_geometry(name)
                    w.add_geometry(name, geometry)
                elif command == 'add':
                    w.add_geometry(name, geometry)

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

from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from droidlet.base_util import to_player_struct, Pos, Look, Player
from droidlet.memory.memory_nodes import PlayerNode
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from agents.loco_mc_agent import LocoMCAgent
from agents.argument_parser import ArgumentParser
from droidlet.memory.robot.loco_memory import LocoAgentMemory, DetectedObjectNode
from droidlet.perception.robot import Perception
from self_perception import SelfPerception
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

    while True:
        base_state = mover.get_base_pos_in_canonical_coords()
        print("base_state: ", base_state)
        # print("rgb_depth: ", mover.get_rgb_depth())
        sio.emit("image_settings", log_settings)
        resolution = log_settings["image_resolution"]
        quality = log_settings["image_quality"]

        rgb_depth = mover.get_rgb_depth()
        serialized_image = rgb_depth.to_struct(resolution, quality)

        sio.emit("rgb", serialized_image["rgb"])
        sio.emit("depth", {
            "depthImg": serialized_image["depth_img"],
            "depthMax": serialized_image["depth_max"],
            "depthMin": serialized_image["depth_min"],
        })

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

        # opcd = opcd.uniform_down_sample(10)
        all_points = np.asarray(opcd.points)
        all_colors = np.asarray(opcd.colors)

        print(all_points.shape)


        if first:
            cmd = 'add'
            first = False
        else:
            cmd = 'replace'
            
        o3dviz.put('pointcloud', cmd, opcd)

        x, y, yaw = base_state.tolist()

        robot_orientation = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=.05,
                                                       cone_radius=.075,
                                                       cylinder_height = .50,
                                                       cone_height = .4,
                                                       resolution=20)
        robot_orientation.compute_vertex_normals()
        robot_orientation.paint_uniform_color([0.1, 0.9, 0.1])
        
        robot_orientation.translate([y, -x, 0], relative=False)
        robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi/2, 0]))
        robot_orientation.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw]))        

        o3dviz.put('bot_orientation', cmd, robot_orientation)

        robot_base = o3d.geometry.TriangleMesh.create_cylinder(radius=.1,
                                                          height=1.,)
        robot_base.translate([y, -x, 0], relative=False)
        robot_base.compute_vertex_normals()
        robot_base.paint_uniform_color([0.9, 0.1, 0.1])

        o3dviz.put('bot_base', cmd, robot_base)
        
        # serialized_pcd = {}
        # serialized_pcd['points'] = points.tolist()
        # serialized_pcd['colors'] = colors.tolist()
        # serialized_pcd['base'] = base_state.tolist()
        # sio.emit("pointcloud", serialized_pcd)
        
        
        time.sleep(0.001)
