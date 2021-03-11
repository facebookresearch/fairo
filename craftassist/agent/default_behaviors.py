"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import os
import sys

import numpy as np
import random
import shape_helpers as sh
import tasks

from mc_util import pos_to_np

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)

from base_agent.dialogue_objects import Say
from ttad.generation_dialogues.generate_utils import prepend_a_an

"""This file contains functions that the agent can perform 
at random when not following player instructions or interacting with the
player"""


def build_random_shape(agent, rand_range=(10, 0, 10), no_chat=False):
    """Pick a random shape from shapes.py and build that"""
    target_loc = agent.pos
    for i in range(3):
        target_loc[i] += np.random.randint(-rand_range[i], rand_range[i] + 1)
    shape = random.choice(sh.SHAPE_NAMES)
    opts = sh.SHAPE_HELPERS[shape]()
    opts["bid"] = sh.bid()
    schematic = sh.SHAPE_FNS[shape](**opts)
    relations = [
        {"pred": "has_name", "obj": shape.lower()},
        {"pred": "has_tag", "obj": shape.lower()},
    ]
    task_data = {
        "blocks_list": schematic,
        "origin": target_loc,
        "verbose": False,
        "schematic_tags": relations,
        "default_behavior": "build_random_shape",  # must == function name. Hacky and I hate it.
    }
    logging.debug("Default behavior: building {}".format(shape))
    agent.memory.task_stack_push(tasks.Build(agent, task_data))

    if not no_chat:
        shape_name = prepend_a_an(shape.lower())
        agent.dialogue_manager.dialogue_stack.append_new(
            Say, "I am building {} while you decide what you want me to do!".format(shape_name)
        )
    return schematic


def come_to_player(agent):
    """Go to where the player is."""
    op = agent.get_other_players()
    if len(op) == 0:
        return
    p = random.choice(agent.get_other_players())
    agent.memory.task_stack_push(tasks.Move(agent, {"target": pos_to_np(p.pos), "approx": 3}))
