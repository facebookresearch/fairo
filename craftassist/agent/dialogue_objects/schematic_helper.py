"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import random
import re
from typing import cast, List, Tuple, Union, Optional

# TODO with subinterpret
from base_agent.dialogue_objects import get_repeat_num
import block_data
import size_words
from .block_helpers import get_block_type
from base_agent.base_util import ErrorWithResponse
from mc_util import Block, most_common_idm

from word2number.w2n import word_to_num
from word_maps import SPECIAL_SHAPE_FNS, SPECIAL_SHAPES_CANONICALIZE


def interpret_shape_schematic(
    interpreter, speaker, d, shapename=None
) -> Tuple[List[Block], List[Tuple[str, str]]]:
    """Return a tuple of 2 values:
    - the schematic blocks, list[(xyz, idm)]
    - a list of (pred, val) tags
    """
    if shapename is not None:
        shape = shapename
    else:
        # For sentences like "Stack" and "Place" that have the shapename in dict
        shape = d["has_shape"]

    numeric_keys = [
        "has_thickness",
        "has_radius",
        "has_depth",
        "has_width",
        "has_height",
        "has_length",
        "has_slope",
        #        "has_orientation", #is this supposed to be numeric key?
        "has_distance",
        "has_base",
    ]

    attrs = {key[4:]: word_to_num(d[key]) for key in numeric_keys if key in d}

    if "has_orientation" in d:
        attrs["orient"] = d["has_orientation"]

    if "has_size" in d:
        attrs["size"] = interpret_size(interpreter, d["has_size"])

    if "has_block_type" in d:
        block_type = get_block_type(d["has_block_type"])
        attrs["bid"] = block_type
    elif "has_colour" in d:
        c = block_data.COLOR_BID_MAP.get(d["has_colour"])
        if c is not None:
            attrs["bid"] = random.choice(c)

    tags = []
    for key, val in d.items():
        if key.startswith("has_"):
            stemmed_val = val
            tags.append((key, stemmed_val))

    return SPECIAL_SHAPE_FNS[shape](**attrs), tags


def interpret_size(interpreter, text) -> Union[int, List[int]]:
    """Processes the has_size_ span value and returns int or list[int]"""
    nums = re.findall("[-0-9]+", text)
    if len(nums) == 1:
        # handle "3", "three", etc.
        return word_to_num(nums[0])
    elif len(nums) > 1:
        # handle "3 x 3", "four by five", etc.
        return [word_to_num(n) for n in nums]
    else:
        # handle "big", "really huge", etc.
        if hasattr(interpreter.agent, "size_str_to_int"):
            return interpreter.agent.size_str_to_int(text)
        else:
            return size_words.size_str_to_int(text)


def interpret_named_schematic(
    interpreter, speaker, d
) -> Tuple[List[Block], Optional[str], List[Tuple[str, str]]]:
    """Return a tuple of 3 values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """
    if "has_name" not in d:
        raise ErrorWithResponse("I don't know what you want me to build.")
    name = d["has_name"]
    stemmed_name = name
    shapename = SPECIAL_SHAPES_CANONICALIZE.get(name) or SPECIAL_SHAPES_CANONICALIZE.get(
        stemmed_name
    )
    if shapename:
        shape_blocks, tags = interpret_shape_schematic(
            interpreter, speaker, d, shapename=shapename
        )
        return shape_blocks, None, tags

    schematic = interpreter.memory.get_schematic_by_name(name)
    if schematic is None:
        schematic = interpreter.memory.get_schematic_by_name(stemmed_name)
        if schematic is None:
            raise ErrorWithResponse("I don't know what you want me to build.")
    tags = [(p, v) for (_, p, v) in interpreter.memory.get_triples(subj=schematic.memid)]
    blocks = schematic.blocks
    # TODO generalize to more general block properties
    # Longer term: remove and put a call to the modify model here
    if d.get("has_colour"):
        old_idm = most_common_idm(blocks.values())
        c = block_data.COLOR_BID_MAP.get(d["has_colour"])
        if c is not None:
            new_idm = random.choice(c)
            for l in blocks:
                if blocks[l] == old_idm:
                    blocks[l] = new_idm
    return list(blocks.items()), schematic.memid, tags


def interpret_schematic(
    interpreter, speaker, d, repeat_dict=None
) -> List[Tuple[List[Block], Optional[str], List[Tuple[str, str]]]]:
    """Return a list of 3-tuples, each with values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """
    # hack, fixme in grammar/standardize.  sometimes the repeat is a sibling of action
    if repeat_dict is not None:
        repeat = cast(int, get_repeat_num(repeat_dict))
    else:
        repeat = cast(int, get_repeat_num(d))
    assert type(repeat) == int, "bad repeat={}".format(repeat)
    if "has_shape" in d:
        blocks, tags = interpret_shape_schematic(interpreter, speaker, d)
        return [(blocks, None, tags)] * repeat
    else:
        return [interpret_named_schematic(interpreter, speaker, d)] * repeat


def get_repeat_dir(d):
    if "repeat" in d:
        direction_name = d.get("repeat", {}).get("repeat_dir", "FRONT")
    elif "schematic" in d:
        direction_name = d["schematic"].get("repeat", {}).get("repeat_dir", "FRONT")
    else:
        direction_name = None
    return direction_name
