"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file has functions to implement different dances for the agent.
"""
import time
import math

from .tasks import Move, Dance

konami_dance = [
    {"translate": (0, 1, 0)},
    {"translate": (0, 1, 0)},
    {"translate": (0, -1, 0)},
    {"translate": (0, -1, 0)},
    {"translate": (0, 0, -1)},
    {"translate": (0, 0, 1)},
    {"translate": (0, 0, -1)},
    {"translate": (0, 0, 1)},
]

# TODO relative to current
head_bob = [
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 90)},
    {"head_yaw_pitch": (0, 0)},
    {"head_yaw_pitch": (0, 0)},
]


def add_default_dances(memory):
    memory.add_dance(
        generate_sequential_move_fn(konami_dance), name="konami_dance", tags=["dance"]
    )
    memory.add_dance(generate_sequential_move_fn(head_bob), name="head_bob", tags=["dance"])


def generate_sequential_move_fn(sequence):
    def move_fn(danceObj, agent):
        if danceObj.tick >= len(sequence):
            return None
        else:
            if danceObj.dance_location is not None and danceObj.tick == 0:
                mv = Move(agent, {"target": danceObj.dance_location, "approx": 0})
                danceObj.dance_location = None
            else:
                mv = Dance(agent, sequence[danceObj.tick])
                danceObj.tick += 1
        return mv

    return move_fn
