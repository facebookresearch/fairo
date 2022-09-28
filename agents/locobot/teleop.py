import os
import sys
import subprocess
import time
import signal
import random
import logging
import faulthandler
import threading
import functools
import cv2
import matplotlib.pyplot as plt

from droidlet import dashboard
from droidlet.dashboard.o3dviz import O3DViz
import numpy as np
from scipy.spatial import distance
import open3d as o3d
from droidlet.lowlevel.hello_robot.remote.obstacle_utils import get_points_in_front, is_obstacle, get_o3d_pointcloud, get_ground_plane

import copy
import math
import json
import time


def get_data(mover):
    cam_transform = mover.bot.get_camera_transform().value
    _, _, rot, trans = mover.cam.get_pcd_data(rotate=False)
    base_state = mover.bot.get_base_state().value
    rgb_depth = mover.get_rgb_depth()
    return cam_transform, rot, trans, base_state, rgb_depth


def save_data(cam_transform, rot, trans, base_state, rgb_depth, nb_saved_data):    
    main_folder = 'sept24_fremont/apartment1_beg_toilet/modular_seal/trajectory'
    step_folder = os.path.join(main_folder, 'step'+str(nb_saved_data))
    os.makedirs(step_folder)
    
    # Frame data (rgb-d and pcd)
    frames_folder = os.path.join(step_folder, 'frames')
    os.makedirs(frames_folder)

    np.save(os.path.join(frames_folder, 'depth.npy'), rgb_depth.depth)    
    cv2.imwrite(os.path.join(frames_folder, 'rgb.png'), rgb_depth.rgb[..., ::-1])
    
    # logs data
    logs_data = {}    
    logs_data['rot_'] = rot.tolist()
    logs_data['trans_'] = trans.tolist()
    logs_data['base_state_'] = base_state.tolist()

    with open(os.path.join(step_folder, 'logs.json'), 'w') as fp:
        json.dump(logs_data, fp)


if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()
    if sys.platform == "darwin":
        webrtc_streaming=False
    else:
        webrtc_streaming=True
    o3dviz = O3DViz(webrtc_streaming)
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
# from agents.locobot.end_to_end_semantic_scout import EndToEndSemanticScout

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()


mover = None

# TODO Cleaner way to get scout object state (semantic map + goal) in dashboard
end_to_end_vis = None
modular_vis = None


@sio.on("sendCommandToAgent")
def get_command(sid, command):
    tokens = command.split()
    command, value = tokens[0], " ".join(tokens[1:])
    if len(value) == 0:
        value = None
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
    global end_to_end_vis
    global modular_vis

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
        
        # Commands we introduce
        elif command == "SEARCH_OBJECT_MODULAR_LEARNED":
            if "_" in value.strip():
                object_goal, episode_id = [x.strip() for x in value.split("_")]
            else:
                object_goal = episode_id = value.strip()
            print("action: SEARCH_OBJECT_MODULAR_LEARNED", object_goal)
            mover.move_to_object(
                object_goal,
                episode_id=episode_id,
                exploration_method="learned",
                blocking=False
            )
            modular_vis = True
            sync()
        elif command == "SEARCH_OBJECT_MODULAR_HEURISTIC":
            if "_" in value.strip():
                object_goal, episode_id = [x.strip() for x in value.split("_")]
            else:
                object_goal = episode_id = value.strip()
            print("action: SEARCH_OBJECT_MODULAR_HEURISTIC", object_goal)
            mover.move_to_object(
                object_goal,
                episode_id=episode_id,
                exploration_method="frontier",
                blocking=False
            )
            modular_vis = True
            sync()
        elif command == "SEARCH_OBJECT_END_TO_END":
            if "_" in value.strip():
                object_goal, episode_id = [x.strip() for x in value.split("_")]
            else:
                object_goal = episode_id = value.strip()
            print("action: SEARCH_OBJECT_END_TO_END", object_goal)
            mover.slam.disable_semantic_map_update()
            scout = EndToEndSemanticScout(
                mover,
                episode_id=episode_id,
                object_goal=object_goal,
                # policy="robot_camera_settings_without_noise_and_coco_detector_il",  # NO DEPTH NOISE IL
                # policy="robot_camera_settings_without_noise_and_coco_detector_rl",  # NO DEPTH NOISE RL
                policy="robot_camera_settings_and_coco_detector_rl",                # WITH DEPTH NOISE 
                # policy="original_camera_settings_and_mp3d_detector_rl",             # ORIGINAL      
            )
            while not scout.finished:
                scout.step(mover)
                end_to_end_vis = scout.semantic_frame
        
        # Active learning commands
        elif command == "ACTIVE_LEARNING_EXPLOR_HEURISTIC":
            print("action: ACTIVE_LEARNING_EXPLOR_HEURISTIC")
            mover.collect_data(
                episode_id=str(value),
                exploration_method="frontier",
                blocking=False
            )
            modular_vis = True
            sync()
        elif command == "ACTIVE_LEARNING_EXPLOR_LEARNED":
            print("action: ACTIVE_LEARNING_EXPLOR_LEARNED")
            mover.collect_data(
                episode_id=str(value),
                exploration_method="learned",
                blocking=False
            )
            modular_vis = True
            sync()
        elif command == "ACTIVE_LEARNING_SEAL":
            print("action: ACTIVE_LEARNING_SEAL")
            mover.collect_data(
                episode_id=str(value),
                exploration_method="seal",
                blocking=False
            )
            modular_vis = True
            sync()
        
        elif command == "LOOK_AT":
            xyz = value.split(',')
            xyz = [float(p) for p in xyz]
            print("action: LOOK_AT", xyz)
            mover.look_at(xyz, turn_base=False)
        elif command == "RESET":
            mover.bot.set_tilt(0.)
            mover.bot.set_pan(0.)
        elif command == "TAKE_PHOTO":
            filename = value.strip()
            # rgb_depth = mover.get_rgb_depth()
            # rgb, depth = rgb_depth.rgb, rgb_depth.depth
            rgb, depth = mover.get_rgb_depth_optimized_for_habitat_transfer()
            plt.imsave(f"pictures/{filename}_rgb.png", rgb)
            plt.imsave(f"pictures/{filename}_depth.png", depth)
            np.save(f"pictures/{filename}_rgb.npy", rgb)
            np.save(f"pictures/{filename}_depth.npy", depth)

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
    if backend == 'habitat':
        mover.bot.set_pan(0.0)
    else: # hellorobot
        mover.bot.set_pan(0.0)
        mover.bot.set_tilt(math.radians(90))
        mover.bot.set_tilt(math.radians(90))
        mover.bot.set_tilt(math.radians(90))
        mover.bot.set_tilt(math.radians(90))
        mover.bot.set_tilt(math.radians(90))
        
        time.sleep(10)
        print('tilt changed.... ready to start!')

    # Whether to save data here or not
    saving_data = True
    if saving_data:
        last_pose = None
        nb_saved_data = 0
    while True:
        counter += 1
        iter_time = time.time_ns() - start_time
        if float(iter_time) / 1e9 > fps_freq :
            # print("FPS: ", round(counter / (float(iter_time) / 1e9), 1), "  ", int(iter_time / 1e6 / counter), "ms")
            counter = 0
            start_time = time.time_ns()
        
        ######## Get robot data
        if saving_data:
            cam_transform_, rot_, trans_, base_state_, rgb_depth_ = get_data(mover)
            base_state_ = np.array(base_state_)
            
            # Save data only if pose changed
            if last_pose is None or np.linalg.norm(last_pose[:2]-base_state_[:2]) >= 0.02 or abs(last_pose[2]-base_state_[2]) >= 0.02:
                save_data(cam_transform_, rot_, trans_, base_state_, rgb_depth_, nb_saved_data)
                last_pose = copy.deepcopy(base_state_)
                nb_saved_data += 1
        ########
        
        # not visualizing when saving data in order to save time
        else:
            base_state = mover.get_base_pos_in_canonical_coords()

            sio.emit("image_settings", log_settings)
            resolution = log_settings["image_resolution"]
            quality = log_settings["image_quality"]

            # this goes from 21ms to 120ms
            rgb_depth = mover.get_rgb_depth()

            points, colors = rgb_depth.ptcloud.reshape(-1, 3), rgb_depth.rgb.reshape(-1, 3)
            colors = colors / 255.

            # TODO Temporary hack to get semantic map in dashboard
            if end_to_end_vis is not None:
                rgb_depth.rgb = end_to_end_vis[:, :, [2, 1, 0]]
            elif modular_vis is not None:
                semantic_map_vis = mover.nav.get_last_semantic_map_vis()
                semantic_map_vis.wait()
                rgb_depth.rgb = semantic_map_vis.value[:, :, [2, 1, 0]]

            # this takes about 1.5 to 2 fps
            serialized_image = rgb_depth.to_struct(resolution, quality)

            sio.emit("rgb", serialized_image["rgb"])
            sio.emit("depth", {
                "depthImg": serialized_image["depth_img"],
                "depthMax": serialized_image["depth_max"],
                "depthMin": serialized_image["depth_min"],
            })

            if all_points is None:
                all_points = points
                all_colors = colors
            else:
                all_points = np.concatenate((all_points, points), axis=0)
                all_colors = np.concatenate((all_colors, colors), axis=0)

            opcd = o3d.geometry.PointCloud()
            opcd.points = o3d.utility.Vector3dVector(all_points)
            opcd.colors = o3d.utility.Vector3dVector(all_colors)
            opcd = opcd.voxel_down_sample(0.03)

            all_points = np.asarray(opcd.points)
            all_colors = np.asarray(opcd.colors)
            
            o3dviz.put('pointcloud', opcd)

            # Plot the robot
            x, y, yaw = base_state.tolist()

            if backend == 'locobot':
                height = 0.63
            else: # hello-robot
                height = 1.41
            o3dviz.add_robot(base_state, height)
            
            sio.emit(
                "map",
                {"x": x, "y": y, "yaw": yaw, "map": mover.get_obstacles_in_canonical_coords()},
            )

        time.sleep(0.001)
