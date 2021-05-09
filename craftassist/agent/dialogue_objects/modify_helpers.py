"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from .. import rotation
from ..shape_transforms import (
    scale,
    thicker,
    shrink_sample,
    replace_by_blocktype,
    replace_by_halfspace,
    fill_flat,
    hollow,
    rotate,
    maybe_convert_to_list,
    maybe_convert_to_npy,
)
from base_agent.base_util import ErrorWithResponse
from base_agent.dialogue_objects import interpret_relative_direction
from .block_helpers import get_block_type


# TODO lots of reuse with build here....
# TODO don't destroy then build if its unecessary...


def handle_rigidmotion(interpreter, speaker, modify_dict, obj):
    old_blocks = list(obj.blocks.items())
    mx, my, mz = np.min([l for l, idm in old_blocks], axis=0)
    angle = modify_dict.get("categorical_angle")
    mirror = modify_dict.get("mirror")
    no_change = False
    if angle or mirror:
        angle = angle or 0
        angle = {0: 0, "LEFT": -90, "RIGHT": 90, "AROUND": 180}[angle]
        if mirror:
            mirror = 0
        else:
            mirror = -1
        new_schematic = maybe_convert_to_list(rotate(old_blocks, angle, mirror))
    else:
        no_change = True
        new_schematic = old_blocks
    location_d = modify_dict.get("location")
    if location_d:
        mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, location_d)
        steps, reldir = interpret_relative_direction(interpreter, location_d)
        origin, _ = interpreter.subinterpret["specify_locations"](
            interpreter, speaker, mems, steps, reldir
        )
    else:
        origin = (mx, my, mz)

    if no_change and origin == (mx, my, mz):
        return None, None

    destroy_task_data = {"schematic": old_blocks}

    # FIXME deal with tags!!!
    build_task_data = {
        "blocks_list": new_schematic,
        "origin": origin,
        #        "schematic_tags": tags,
    }

    return destroy_task_data, build_task_data


# TODO don't destroy the whole thing, just the extra blocks
def handle_scale(interpreter, speaker, modify_dict, obj):
    old_blocks = list(obj.blocks.items())
    bounds = obj.get_bounds()
    mx, my, mz = (bounds[0], bounds[2], bounds[4])
    csf = modify_dict.get("categorical_scale_factor")
    origin = [mx, my, mz]
    if not csf:
        if modify_dict.get("numerical_scale_factor"):
            raise ErrorWithResponse("I don't know how to handle numerical_scale_factor yet")
        else:
            raise ErrorWithResponse(
                "I think I am supposed to scale something but I don't know which dimensions to scale"
            )
    destroy_task_data = {"schematic": old_blocks}
    if csf == "WIDER":
        if bounds[1] - bounds[0] > bounds[5] - bounds[4]:
            lam = (2.0, 1.0, 1.0)
        else:
            lam = (1.0, 1.0, 2.0)
        new_blocks = maybe_convert_to_list(scale(old_blocks, lam))
        destroy_task_data = None
    elif csf == "NARROWER":
        if bounds[1] - bounds[0] > bounds[5] - bounds[4]:
            lam = (0.5, 1.0, 1.0)
        else:
            lam = (1.0, 1.0, 0.5)
        new_blocks = maybe_convert_to_list(shrink_sample(old_blocks, lam))
    elif csf == "TALLER":
        lam = (1.0, 2.0, 1.0)
        new_blocks = maybe_convert_to_list(scale(old_blocks, lam))
        destroy_task_data = None
    elif csf == "SHORTER":
        lam = (1.0, 0.5, 1.0)
        new_blocks = maybe_convert_to_list(shrink_sample(old_blocks, lam))
    elif csf == "SKINNIER":
        lam = (0.5, 1.0, 0.5)
        new_blocks = maybe_convert_to_list(shrink_sample(old_blocks, lam))
    elif csf == "FATTER":
        lam = (2.0, 1.0, 2.0)
        new_blocks = maybe_convert_to_list(scale(old_blocks, lam))
        destroy_task_data = None
    elif csf == "BIGGER":
        lam = (2.0, 2.0, 2.0)
        destroy_task_data = None
        new_blocks = maybe_convert_to_list(scale(old_blocks, lam))
    elif csf == "SMALLER":
        lam = (0.5, 0.5, 0.5)
        new_blocks = maybe_convert_to_list(shrink_sample(old_blocks, lam))

    M_new = np.max([l for l, idm in new_blocks], axis=0)
    m_new = np.min([l for l, idm in new_blocks], axis=0)
    new_extent = (M_new[0] - m_new[0], M_new[1] - m_new[1], M_new[2] - m_new[2])
    old_extent = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    origin = (
        mx - (new_extent[0] - old_extent[0]) // 2,
        my,
        mz - (new_extent[2] - old_extent[2]) // 2,
    )

    # FIXME deal with tags!!!
    build_task_data = {
        "blocks_list": new_blocks,
        "origin": origin,
        #        "schematic_tags": tags,
    }
    return destroy_task_data, build_task_data


def handle_fill(interpreter, speaker, modify_dict, obj):
    old_blocks = list(obj.blocks.items())
    bounds = obj.get_bounds()
    mx, my, mz = (bounds[0], bounds[2], bounds[4])
    origin = [mx, my, mz]
    destroy_task_data = None
    if modify_dict.get("modify_type") == "FILL":
        if modify_dict.get("new_block"):
            # TODO FILTERS, also in build
            block_type = get_block_type(modify_dict["new_block"])
            new_blocks = fill_flat(old_blocks, fill_material=block_type)
        else:
            new_blocks = fill_flat(old_blocks)
    else:
        # modify_dict.get("modify_type") == "hollow"
        new_blocks = hollow(old_blocks)

    #    destroy_task_data = {"schematic": old_blocks}
    # FIXME deal with tags!!!
    build_task_data = {
        "blocks_list": maybe_convert_to_list(new_blocks),
        "origin": origin,
        #        "schematic_tags": tags,
    }
    return destroy_task_data, build_task_data


def handle_replace(interpreter, speaker, modify_dict, obj):
    old_blocks = list(obj.blocks.items())
    bounds = obj.get_bounds()
    mx, my, mz = (bounds[0], bounds[2], bounds[4])
    origin = (mx, my, mz)
    new_block_type = get_block_type(modify_dict["new_block"])
    destroy_task_data = None
    if modify_dict.get("old_block"):
        # TODO FILTERS, also in build
        # TODO "make the red blocks green" etc- currently get_block type does not return a list of possibilities
        old_block_type = get_block_type(modify_dict["old_block"])
        new_blocks = replace_by_blocktype(
            old_blocks, new_idm=new_block_type, current_idm=old_block_type
        )
    else:
        geom_d = modify_dict.get("replace_geometry")
        geometry = {}
        schematic = maybe_convert_to_npy(old_blocks)
        geometry["offset"] = np.array(schematic.shape[:3]) / 2
        reldir = geom_d.get("relative_direction", "TOP")
        if reldir == "TOP":
            reldir = "UP"
        elif reldir == "BOTTOM":
            reldir = "DOWN"
        reldir_vec = rotation.DIRECTIONS[reldir]
        look = (
            interpreter.agent.perception_modules["low_level"]
            .get_player_struct_by_name(speaker)
            .look
        )
        dir_vec = rotation.transform(reldir_vec, look.yaw, 0, inverted=True)
        geometry["v"] = dir_vec
        projections = []
        for l, idm in old_blocks:
            projections.append((np.array(l) - geometry["offset"]) @ reldir_vec)
        a = geom_d.get("amount", "HALF")
        if a == "QUARTER":
            geometry["threshold"] = (np.max(projections) - np.min(projections)) / 4
        else:
            geometry["threshold"] = 0.0
        new_blocks = replace_by_halfspace(old_blocks, new_idm=new_block_type, geometry=geometry)

    # FIXME deal with tags!!!
    build_task_data = {
        "blocks_list": maybe_convert_to_list(new_blocks),
        "origin": origin,
        #        "schematic_tags": tags,
    }
    return destroy_task_data, build_task_data


# TODO don't destroy first
def handle_thicken(interpreter, speaker, modify_dict, obj):
    old_blocks = list(obj.blocks.items())
    bounds = obj.get_bounds()
    mx, my, mz = (bounds[0], bounds[2], bounds[4])
    origin = [mx, my, mz]
    if modify_dict.get("modify_type") == "THICKER":
        num_blocks = modify_dict.get("num_blocks", 1)
        new_blocks = thicker(old_blocks, delta=num_blocks)
    else:
        raise ErrorWithResponse("I don't know how thin out blocks yet")

    destroy_task_data = {"schematic": old_blocks}
    # FIXME deal with tags!!!
    build_task_data = {
        "blocks_list": new_blocks,
        "origin": origin,
        #        "schematic_tags": tags,
    }
    return destroy_task_data, build_task_data
