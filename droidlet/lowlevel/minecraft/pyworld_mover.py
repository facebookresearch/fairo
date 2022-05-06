"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import time
import socketio
from droidlet.base_util import XYZ, Pos, Look
from droidlet.shared_data_struct.craftassist_shared_utils import Player, Item, ItemStack, Mob


class DataCallback:
    def __call__(self, data):
        self.data = data


def wait_for_data(data_callback, step=0.001, max_steps=100):
    data = None
    count = 0
    while not data:
        data = getattr(data_callback, "data", None)
        if data is not None or count > max_steps:
            return data
        count += 1
        time.sleep(step)


class PyWorldMover:
    def __init__(self, port=25565, ip="localhost"):
        sio = socketio.Client()
        try:
            sio.connect("http://{}:{}".format(ip, port))
            time.sleep(0.1)
            sio.emit("init_player", {"player_type": "agent"})
        except:
            raise Exception("unable to connect to server on port {} at ip {}".format(port, ip))

        print("connected to server on port {} at ip {}".format(port, ip))
        self.sio = sio

    def get_line_of_sight(self):
        D = DataCallback()
        self.sio.emit("line_of_sight", {}, callback=D)
        return wait_for_data(D)

    def set_look(self, yaw, pitch):
        self.sio.emit("set_look", {"yaw": yaw, "pitch": pitch})

    def get_player(self):
        D = DataCallback()
        self.sio.emit("get_player", callback=D)
        return wait_for_data(D)
