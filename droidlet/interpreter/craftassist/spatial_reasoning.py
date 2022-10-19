"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np

import droidlet.base_util
from droidlet.shared_data_struct import rotation
from droidlet.perception.craftassist import heuristic_perception
from droidlet.base_util import to_block_center, to_block_pos
from droidlet.shared_data_struct.craftassist_shared_utils import arrange
from droidlet.shared_data_structs import ErrorWithResponse

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
        repeat_num = max(repeat_num, len(objects))
        _, result_mem = interpreter.memory.basic_search(
            f"SELECT MEMORY FROM ReferenceObject WHERE ref_type=player AND name={speaker}"
        )
        player_mem = result_mem[0] if len(result_mem) == 1 else None
        get_locs_from_entity = interpreter.get_locs_from_entity
        origin = compute_location_heuristic(player_mem, mems, steps, reldir, get_locs_from_entity)
        if repeat_num > 1:
            schematic = None if len(objects) == 0 else objects[0][0]
            offsets = get_repeat_arrangement(
                player_mem, repeat_num, repeat_dir, mems, schematic, padding
            )
        else:
            offsets = [(0, 0, 0)]
        origin = post_process_loc(origin, interpreter)
        offsets = [post_process_loc(o, interpreter) for o in offsets]
        return origin, offsets


# There will be at least one mem in mems
def compute_location_heuristic(player_mem, mems, steps, reldir, get_locs_from_entity):
    loc = mems[0].get_pos()
    if reldir is not None:
        steps = steps or DEFAULT_NUM_STEPS
        if reldir == "BETWEEN":
            loc = (np.add(mems[0].get_pos(), mems[1].get_pos())) / 2
            loc = (loc[0], loc[1], loc[2])
        elif reldir == "INSIDE":
            for i in range(len(mems)):
                mem = mems[i]
                # FIXME
                locs = heuristic_perception.find_inside(mem, get_locs_from_entity)
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
            yaw, _ = player_mem.get_yaw_pitch()
            dir_vec = rotation.transform(reldir_vec, yaw, 0, inverted=True)
            loc = steps * np.array(dir_vec) + to_block_center(loc)
    elif steps is not None:
        loc = to_block_center(loc) + [0, 0, steps]
    return to_block_pos(loc)


def get_repeat_arrangement(
    player_mem, repeat_num, repeat_dir, ref_mems, schematic=None, padding=(1, 1, 1)
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

        offsets = arrange("circle", schematic, shapeparams)
    else:
        reldir_vec = rotation.DIRECTIONS[repeat_dir]
        # this should be an inverse transform so we set inverted=True
        yaw, _ = player_mem.get_yaw_pitch()
        dir_vec = rotation.transform(reldir_vec, yaw, 0, inverted=True)
        max_ind = np.argmax(dir_vec)
        shapeparams["extra_space"] = padding[max_ind]
        shapeparams["orient"] = dir_vec
        offsets = arrange("line", schematic, shapeparams)
    offsets = [tuple(to_block_pos(o)) for o in offsets]
    return offsets
