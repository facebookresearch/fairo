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
        
def wait_for_data(data_callback, step=.001, max_steps = 100):
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
            sio.connect('http://{}:{}'.format(ip, port))
            time.sleep(.1)
            sio.emit("init_player", {"player_type": "agent"})
        except:
            raise Exception("unable to connect to server on port {} at ip {}".format(port, ip)) 

        print("connected to server on port {} at ip {}".format(port, ip))
        self.sio = sio

    def set_look(self, yaw, pitch):
        self.sio.emit("set_look", {"yaw": yaw, "pitch":pitch})

    def step_pos_x(self):
        self.sio.emit("rel_move", {"x": 1})

    def step_neg_x(self):
        self.sio.emit("rel_move", {"x": -1})

    def step_pos_y(self):
        self.sio.emit("rel_move", {"y": 1})

    def step_neg_y(self):
        self.sio.emit("rel_move", {"y": -1})

    def step_pos_z(self):
        self.sio.emit("rel_move", {"z": 1})

    def step_neg_z(self):
        self.sio.emit("rel_move", {"z": -1})

    def step_forward(self):
        self.sio.emit("rel_move", {"forward": 1})

    def set_held_item(self, idm):
        self.sio.emit("set_held_item", {"idm": idm})

    def dig(self, x, y, z):
        D = DataCallback()
        self.sio.emit("dig", {"loc": [x, y, z]}, callback=D)
        # return True if the world says the block was dug, False otherwise
        placed = wait_for_data(D)
        # this is sketchy: if world doesn't respond in time, block placement will
        # show False even if block was placed....  TODO? check with a get_blocks?
        if placed is None:
            placed = False
        return placed
        
    def place_block(self, x, y, z):
        """ place the block in mainhand.  does nothing if mainhand empty"""
        D = DataCallback()
        self.sio.emit("place_block", {"loc": [x, y, z]}, callback=D)
        # return True if the world says the block was placed, False otherwise
        placed = wait_for_data(D)
        # this is sketchy: if world doesn't respond in time, block placement will
        # show False even if block was placed....  TODO? check with a get_blocks?
        if placed is None:
            placed = False
        return placed

    def get_player(self):
        D = DataCallback()
        self.sio.emit("get_player", callback=D)
        try:
            eid, name, pos, look, mainhand, _ = wait_for_data(D)["player"]
            if mainhand is not None:
                mainhand = Item(*mainhand)
                return Player(eid, name, Pos(*pos), Look(*look), mainhand)
            else:
                return Player(eid, name, Pos(*pos), Look(*look))
        except:
            return None
        
    def get_line_of_sight(self):
        D = DataCallback()
        self.sio.emit("line_of_sight", {}, callback=D)
        pos = wait_for_data(D)
        if pos is not None:
            pos = pos["pos"]
        return pos

    def get_changed_blocks(self):
        D = DataCallback()
        self.sio.emit("get_changed_blocks", callback=D)
        blocks = wait_for_data(D)
        def decode_str_loc(x):
            return tuple(int(z) for z in x.strip("(").strip(")").split(","))
        if blocks:
            # can't send dicts with tuples for keys :(
            blocks = {decode_str_loc(loc): idm for loc, idm in blocks.items()}
        return blocks


if __name__=="__main__":
    m = PyWorldMover(port=6000)
    m.set_held_item((1,0))
    r = m.place_block(10,10, 10)
    print(r)
    print(m.get_changed_blocks())
    r = m.dig(10,10, 10)
    print(r)
    print(m.get_changed_blocks())
