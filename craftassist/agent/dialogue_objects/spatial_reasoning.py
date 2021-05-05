"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
from .. import rotation
from .. import shapes

from .. import heuristic_perception
from ..mc_util import pos_to_np, to_block_center, to_block_pos, ErrorWithResponse

DEFAULT_NUM_STEPS = 5


def post_process_loc(loc, interpreter):
    return to_block_pos(loc)


class ComputeLocations:
    def __call__(
        self,
        interpreter,
        speaker,
        mems,
        steps,
        reldir,
        repeat_num=1,
        repeat_dir=None,
        objects=[],
        padding=(1, 1, 1),
    ):
        agent = interpreter.agent
        repeat_num = max(repeat_num, len(objects))
        player_look = agent.perception_modules["low_level"].get_player_struct_by_name(speaker).look
        player_pos = pos_to_np(agent.get_player().pos)
        speaker_pos = pos_to_np(
            agent.perception_modules["low_level"].get_player_struct_by_name(speaker).pos
        )
        origin = compute_location_heuristic(player_look, player_pos, mems, steps, reldir)
        if repeat_num > 1:
            schematic = None if len(objects) == 0 else objects[0][0]
            offsets = get_repeat_arrangement(
                player_look, repeat_num, repeat_dir, mems, schematic, padding
            )
        else:
            offsets = [(0, 0, 0)]
        origin = post_process_loc(origin, interpreter)
        offsets = [post_process_loc(o, interpreter) for o in offsets]
        return origin, offsets


# There will be at least one mem in mems
def compute_location_heuristic(player_look, player_pos, mems, steps, reldir):
    loc = mems[0].get_pos()
    if reldir is not None:
        steps = steps or DEFAULT_NUM_STEPS
        if reldir == "BETWEEN":
            loc = (np.add(mems[0].get_pos(), mems[1].get_pos())) / 2
            loc = (loc[0], loc[1], loc[2])
        elif reldir == "INSIDE":
            for i in range(len(mems)):
                mem = mems[i]
                locs = heuristic_perception.find_inside(mem)
                if len(locs) > 0:
                    break
            if len(locs) == 0:
                raise ErrorWithResponse("I don't know how to go inside there")
            else:
                loc = locs[0]
        elif reldir == "NEAR":
            pass
        elif reldir == "AROUND":
            pass
        else:  # LEFT, RIGHT, etc...
            reldir_vec = rotation.DIRECTIONS[reldir]
            # this should be an inverse transform so we set inverted=True
            dir_vec = rotation.transform(reldir_vec, player_look.yaw, 0, inverted=True)
            loc = steps * np.array(dir_vec) + to_block_center(loc)
    elif steps is not None:
        loc = to_block_center(loc) + [0, 0, steps]
    return to_block_pos(loc)


def get_repeat_arrangement(
    player_look, repeat_num, repeat_dir, ref_mems, schematic=None, padding=(1, 1, 1)
):
    shapeparams = {}
    # default repeat dir is LEFT
    if not repeat_dir:
        repeat_dir = "LEFT"
    # eventually fix this to allow number based on shape
    shapeparams["N"] = repeat_num

    if repeat_dir == "AROUND":
        # TODO vertical "around"
        shapeparams["orient"] = "xy"
        shapeparams["extra_space"] = max(padding)
        central_object = ref_mems[0]
        bounds = central_object.get_bounds()
        b = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        shapeparams["encircled_object_radius"] = b

        offsets = shapes.arrange("circle", schematic, shapeparams)
    else:

        reldir_vec = rotation.DIRECTIONS[repeat_dir]
        # this should be an inverse transform so we set inverted=True
        dir_vec = rotation.transform(reldir_vec, player_look.yaw, 0, inverted=True)
        max_ind = np.argmax(dir_vec)
        shapeparams["extra_space"] = padding[max_ind]
        shapeparams["orient"] = dir_vec
        offsets = shapes.arrange("line", schematic, shapeparams)
    offsets = [tuple(to_block_pos(o)) for o in offsets]
    return offsets
