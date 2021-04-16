"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from .utils import Pos, Look
from droidlet.base_util import MOBS_BY_ID

FLAT_GROUND_DEPTH = 8
FALL_SPEED = 1
# TODO make these prettier
MOB_COLORS = {
    "rabbit": (0.3, 0.3, 0.3),
    "cow": (0.9, 0.9, 0.9),
    "pig": (0.9, 0.5, 0.5),
    "chicken": (0.9, 0.9, 0.0),
    "sheep": (0.6, 0.6, 0.6),
}
MOB_META = {101: "rabbit", 92: "cow", 90: "pig", 93: "chicken", 91: "sheep"}
MOB_SPEED = {"rabbit": 1, "cow": 0.3, "pig": 0.5, "chicken": 1, "sheep": 0.3}
MOB_LOITER_PROB = {"rabbit": 0.3, "cow": 0.5, "pig": 0.3, "chicken": 0.1, "sheep": 0.5}
MOB_LOITER_TIME = {"rabbit": 2, "cow": 2, "pig": 2, "chicken": 1, "sheep": 2}
MOB_STEP_HEIGHT = {"rabbit": 1, "cow": 1, "pig": 1, "chicken": 2, "sheep": 2}
MOB_DIRECTION_CHANGE_PROB = {"rabbit": 0.1, "cow": 0.1, "pig": 0.2, "chicken": 0.3, "sheep": 0.2}


class Opt:
    pass


class MobInfo:
    pass


def make_mob_opts(mobname):
    opts = Opt()
    opts.mobname = mobname
    opts.direction_change_prob = MOB_DIRECTION_CHANGE_PROB[mobname]
    opts.color = MOB_COLORS[mobname]
    opts.speed = MOB_SPEED[mobname]
    opts.loiter_prob = MOB_LOITER_PROB[mobname]
    opts.loiter_time = MOB_LOITER_TIME[mobname]
    opts.step_height = MOB_STEP_HEIGHT[mobname]
    opts.mobType = list(MOBS_BY_ID.keys())[list(MOBS_BY_ID.values()).index(mobname)]
    return opts


def check_bounds(p, sl):
    if p >= sl or p < 0:
        return -1
    return 1


class SimpleMob:
    def __init__(self, opts, start_pos=None, start_look=(0.0, 0.0)):
        self.mobname = opts.mobname
        self.color = opts.color
        self.direction_change_prob = opts.direction_change_prob
        self.loiter_prob = opts.loiter_prob
        self.loiter_time = opts.loiter_time
        self.speed = opts.speed
        self.step_height = opts.step_height
        self.pos = start_pos
        self.look = start_look
        self.loitering = -1
        self.new_direction()
        self.entityId = str(np.random.randint(0, 100000))
        self.mobType = opts.mobType
        self.agent_built = False

    def add_to_world(self, world):
        self.world = world
        if self.pos is None:
            xz = np.random.randint(0, world.sl, (2,))
            slice = self.world.blocks[xz[0], :, xz[1], 0]
            nz = np.flatnonzero(slice)
            if len(nz) == 0:
                # mob will be floating, but why no floor here?
                h = 0
            else:
                # if top block is nonzero mob will be trapped
                h = nz[-1]
            off = self.world.coord_shift
            self.pos = (float(xz[0]) + off[0], float(h + 1) + off[1], float(xz[1]) + off[2])
        self.world.mobs.append(self)

    def get_info(self):
        info = MobInfo()
        info.entityId = self.entityId
        info.pos = Pos(*self.pos)
        info.look = Look(*self.look)
        info.mobType = self.mobType
        info.color = self.color
        info.mobname = self.mobname
        return info

    def new_direction(self):
        new_direction = np.random.randn(2)
        self.direction = new_direction / np.linalg.norm(new_direction)
        # self.look ##NOT IMPLEMENTED

    def step(self):
        # check if falling:
        x, y, z = self.world.to_world_coords(self.pos)
        fy = int(np.floor(y))
        rx = int(np.round(x))
        rz = int(np.round(z))
        if y > 0:
            if self.world.blocks[rx, fy - 1, rz, 0] == 0:
                self.pos = (self.pos[0], self.pos[1] - FALL_SPEED, self.pos[2])
                return
        # TODO when look implemented: change looks when loitering
        if self.loitering >= 0:
            self.loitering += 1
            if self.loitering > self.loiter_time:
                self.loitering = -1
            return
        if np.random.rand() < self.loiter_prob:
            self.loitering = 0
            return
        if np.random.rand() < self.direction_change_prob:
            self.new_direction()
        step = self.direction * self.speed
        bx = check_bounds(int(np.round(x + step[0])), self.world.sl)
        bz = check_bounds(int(np.round(z + step[1])), self.world.sl)
        # if hitting boundary, reverse...
        self.direction[0] = bx * self.direction[0]
        self.direction[1] = bz * self.direction[1]
        step = self.direction * self.speed
        new_x = step[0] + x
        new_z = step[1] + z
        # is there a block in new location? if no go there, if yes go up
        for i in range(self.step_height):
            if fy + i >= self.world.sl:
                self.new_direction()
                return
            if self.world.blocks[int(np.round(new_x)), fy + i, int(np.round(new_z)), 0] == 0:
                self.pos = self.world.from_world_coords((new_x, y + i, new_z))
                return
        # couldn't get past a wall of blocks, try a different dir
        self.new_direction()
        return


class LoopMob(SimpleMob):
    def __init__(self, opts, move_sequence):
        # move sequence is a list of ((x, y, z), (yaw, pitch)) tuples
        super().__init__(opts)
        self.move_sequence = move_sequence
        self.pos = move_sequence[0][0]
        self.count = 0

    def step(self):
        c = self.count % len(self.move_sequence)
        self.pos = self.move_sequence[c][0]
        #        print("in loopmob step, pos is " + str(self.pos))
        self.look = self.move_sequence[c][1]
        self.count += 1
