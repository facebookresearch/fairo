"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from collections import namedtuple
import math


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


Properties = namedtuple("properties", ["x", "y", "z"])


class DanceMovement(object):
    def __init__(self, agent, move_fn, dance_location=None):
        self.agent = agent
        self.bot = agent.mover.bot
        self.move_fn = move_fn
        self.dance_location = dance_location
        self.tick = 0

    def wave(self):
        for _ in range(3):
            self.bot.set_joint_positions([0.4, 0.0, -1, 0.5, -0.1], plan=False)
            while not self.bot.command_finished():
                time.sleep(0.5)
            self.bot.set_joint_positions([-0.4, 0.0, -1, -0.5, -0.1])
            while not self.bot.command_finished():
                time.sleep(0.5)
        self.bot.set_joint_positions([0.0, -math.pi / 4.0, math.pi / 2.0, 0.0, 0.0], plan=False)

    def get_move(self):
        # move_fn should output a tuple (dx, dy, dz) corresponding to a
        # change in Movement or None
        # if None then Movement is finished
        # can output
        return self.move_fn(self, self.agent)
