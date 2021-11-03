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

import threading
import queue
import time
import math

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
    value = float(value)
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
    move = float(data['move'])
    yaw = float(data['yaw'])
    velocity = float(data['velocity'])
    global mover
    if mover == None:
        return
    if value is not None:
        move = value

    movement = [0.0, 0.0, 0.0]
    for command in commands:
        if command == "MOVE_FORWARD":
            movement[0] += move
            print("action: FORWARD")
        elif command == "MOVE_BACKWARD":
            movement[0] -= move
            print("action: BACKWARD")
        elif command == "MOVE_LEFT":
            movement[2] += yaw
            print("action: LEFT")
        elif command == "MOVE_RIGHT":
            movement[2] -= yaw
            print("action: RIGHT")
        elif command == "PAN_LEFT":
            mover.bot.set_pan(mover.bot.get_pan().value + yaw)
        elif command == "PAN_RIGHT":
            mover.bot.set_pan(mover.bot.get_pan().value - yaw)
        elif command == "TILT_UP":
            mover.bot.set_tilt(mover.bot.get_tilt().value + yaw)
        elif command == "TILT_DOWN":
            mover.bot.set_tilt(mover.bot.get_tilt().value - yaw)
        elif command == "LOG_DATA":
            mover.log_data(value) # in seconds
        elif command == "STOP_ROBOT":
            mover.stop()
        elif command == "UNSTOP_ROBOT":
            mover.unstop()

        print(command, movement)
        mover.move_relative([movement], blocking=False)

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
        if (time.time() - start_time) > fps_freq * 10 :
            print("base_state: ", base_state)
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

        time.sleep(0.001)
