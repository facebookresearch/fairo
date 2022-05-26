"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import numpy as np
import random
from droidlet.interpreter.craftassist import tasks
from droidlet.dialog.dialogue_task import Say
from droidlet.base_util import prepend_a_an, pos_to_np
from droidlet.memory.memory_nodes import TaskNode
#from droidlet.event import sio

"""This file contains functions that the agent can perform 
at random when not following player instructions or interacting with the
player"""


def build_random_shape(agent, shape_util_dict, rand_range=(10, 0, 10), no_chat=False):
    """Pick a random shape from shapes.py and build that"""
    target_loc = agent.pos
    for i in range(3):
        target_loc[i] += np.random.randint(-rand_range[i], rand_range[i] + 1)
    shape = random.choice(shape_util_dict["shape_names"])
    opts = shape_util_dict["shape_option_fn_map"][shape]()
    opts["bid"] = shape_util_dict["bid"]
    schematic = shape_util_dict["shape_fns"][shape](**opts)
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
        Say(
            agent,
            task_data={
                "response_options": "I am building {} while you decide what you want me to do!".format(
                    shape_name
                )
            },
        )
    #        sio.emit("performDefaultBehavior", {"default_behavior": "build_random_shape",
    #                                            "msg": "I am building {} while you decide what you want me to do!".format(shape_name)})
    return schematic


def come_to_player(agent):
    """Go to where the player is."""
    op = agent.get_other_players()
    if len(op) == 0:
        return
    p = random.choice(agent.get_other_players())
    TaskNode(agent.memory, tasks.Move(agent, {"target": pos_to_np(p.pos), "approx": 3}).memid)


#    sio.emit("performDefaultBehavior", {"default_behavior": "come_to_player",
#                                        "msg": "I am coming to {} while you decide what you want me to do!".format(p.name)})
