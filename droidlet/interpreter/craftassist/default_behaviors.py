"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import numpy as np
import random
from droidlet.interpreter.craftassist import tasks
from droidlet.dialog.dialogue_objects import Say
from droidlet.base_util import prepend_a_an, pos_to_np
from droidlet.memory.memory_nodes import TaskNode

"""This file contains functions that the agent can perform 
at random when not following player instructions or interacting with the
player"""


def build_random_shape(agent, shape_helper_dict, rand_range=(10, 0, 10),  no_chat=False):
    """Pick a random shape from shapes.py and build that"""
    target_loc = agent.pos
    for i in range(3):
        target_loc[i] += np.random.randint(-rand_range[i], rand_range[i] + 1)
    shape = random.choice(shape_helper_dict["shape_names"])
    opts = shape_helper_dict["shape_helper"][shape]()
    opts["bid"]  = shape_helper_dict["bid"]
    schematic = shape_helper_dict["shape_fns"][shape](**opts)
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
    TaskNode(agent.memory, tasks.Build(agent, task_data).memid)

    if not no_chat:
        shape_name = prepend_a_an(shape.lower())
        # FIXME agent , also don't push directly to stack, ask the manager?
        agent.memory.dialogue_stack_append_new(
            Say, "I am building {} while you decide what you want me to do!".format(shape_name)
        )
    return schematic


def come_to_player(agent):
    """Go to where the player is."""
    op = agent.get_other_players()
    if len(op) == 0:
        return
    p = random.choice(agent.get_other_players())
    TaskNode(agent.memory, tasks.Move(agent, {"target": pos_to_np(p.pos), "approx": 3}).memid)
