"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import random
import re
import numpy as np
from typing import cast, List, Tuple, Union, Optional

# TODO with subinterpret
from base_agent.dialogue_objects import get_repeat_num
import block_data
import size_words
from .block_helpers import get_block_type
from base_agent.base_util import ErrorWithResponse, number_from_span
from mc_util import Block, most_common_idm

from word2number.w2n import word_to_num
from word_maps import SPECIAL_SHAPE_FNS, SPECIAL_SHAPES_CANONICALIZE


def get_properties_from_triples(triples_list, p):
    return [x.get("obj_text") for x in triples_list if p in x.values()]


def get_attrs_from_triples(triples, interpreter):
    numeric_keys = {
        "has_thickness": get_properties_from_triples(triples, "has_thickness"),
        "has_radius": get_properties_from_triples(triples, "has_radius"),
        "has_depth": get_properties_from_triples(triples, "has_depth"),
        "has_width": get_properties_from_triples(triples, "has_width"),
        "has_height": get_properties_from_triples(triples, "has_height"),
        "has_length": get_properties_from_triples(triples, "has_length"),
        "has_slope": get_properties_from_triples(triples, "has_slope"),
        "has_distance": get_properties_from_triples(triples, "has_distance"),
        "has_base": get_properties_from_triples(triples, "has_base"),
    }

    attrs = {key[4:]: word_to_num(val[0]) for key, val in numeric_keys.items() if any(val)}

    text_keys = {
        "has_orientation": get_properties_from_triples(triples, "has_orientation"),
        "has_size": get_properties_from_triples(triples, "has_size"),
        "has_block_type": get_properties_from_triples(triples, "has_block_type"),
        "has_colour": get_properties_from_triples(triples, "has_colour"),
    }

    if any(text_keys["has_orientation"]):
        attrs["orient"] = text_keys["has_orientation"][0]

    if any(text_keys["has_size"]):
        attrs["size"] = interpret_size(interpreter, text_keys["has_size"][0])

    if any(text_keys["has_block_type"]):
        block_type = get_block_type(text_keys["has_block_type"][0])
        attrs["bid"] = block_type
    elif any(text_keys["has_colour"]):
        c = block_data.COLOR_BID_MAP.get(text_keys["has_colour"][0])
        if c is not None:
            attrs["bid"] = random.choice(c)

    return attrs


# FIXME merge with shape_schematic
# FIXME we should be able to do fancy stuff here, like fill the x with (copies of) schematic y
def interpret_fill_schematic(
    interpreter, speaker, d, hole_locs, hole_idm
) -> Tuple[List[Block], List[Tuple[str, str]]]:
    """Return a tuple of 2 values:
    - the schematic blocks, list[(xyz, idm)]
    - a list of (pred, val) tags

    the "hole" input is a list of xyz coordinates giving a "mold" to be filled.
    """

    filters_d = d.get("filters", {})
    triples = filters_d.get("triples", [])
    attrs = get_attrs_from_triples(triples, interpreter)

    h = attrs.get("height") or attrs.get("depth") or attrs.get("thickness")
    bid = attrs.get("bid") or hole_idm or (1, 0)
    origin = np.min(hole_locs, axis=0)
    ymin = origin[1]
    if h:
        blocks_list = [((x, y, z), bid) for (x, y, z) in hole_locs if y - ymin < h]
    else:
        blocks_list = [((x, y, z), bid) for (x, y, z) in hole_locs]
    tags = []
    for t in triples:
        key = t.get("pred_text", "")
        if key.startswith("has_"):
            val = t.get("obj_text", "")
            stemmed_val = val
            if val:
                tags.append((key, stemmed_val))

    return blocks_list, tags


def interpret_shape_schematic(
    interpreter, speaker, d, shapename=None
) -> Tuple[List[Block], List[Tuple[str, str]]]:
    """Return a tuple of 2 values:
    - the schematic blocks, list[(xyz, idm)]
    - a list of (pred, val) tags

    warning:  if multiple possibilities are given for the same tag, current
    heursitic just picks one.  e.g. if the lf is
        "triples" : [{"pred_text": "has_colour", "obj_text": "red"},
                     {"pred_text": "has_colour", "obj_text": "blue"}]
    will currently just pick red.   Same for other properties encoded in triples
    """
    # FIXME this is not compositional, and does not properly use FILTERS
    filters_d = d.get("filters", {})
    triples = filters_d.get("triples", [{"pred_text": "has_shape", "obj_text": "cube"}])
    if shapename is not None:
        shape = shapename
    else:
        # For sentences like "Stack" and "Place" that have the shapename in dict
        shapes = get_properties_from_triples(triples, "has_shape")
        if any(shapes):
            # see warning above w.r.t. 0
            shape = shapes[0]

    attrs = get_attrs_from_triples(triples, interpreter)

    tags = []
    for t in triples:
        key = t.get("pred_text", "")
        if key.startswith("has_"):
            val = t.get("obj_text", "")
            stemmed_val = val
            if val:
                tags.append((key, stemmed_val))

    return SPECIAL_SHAPE_FNS[shape.upper()](**attrs), tags


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

    warning:  if multiple possibilities are given for the same tag, current
    heursitic just picks one.  e.g. if the lf is
        "triples" : [{"pred_text": "has_colour", "obj_text": "red"},
                     {"pred_text": "has_colour", "obj_text": "blue"}]
    will currently just pick red.   Same for other properties encoded in triples
    """
    # FIXME! this is not compositional, and is not using full FILTERS handlers
    filters_d = d.get("filters", {})
    triples = filters_d.get("triples", [])
    names = get_properties_from_triples(triples, "has_name")
    if not any(names):
        raise ErrorWithResponse("I don't know what you want me to build.")
    name = names[0]
    stemmed_name = name.strip("s")  # why aren't we using stemmer anymore?
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
    colours = get_properties_from_triples(triples, "has_colour")
    if any(colours):
        colour = colours[0]
        old_idm = most_common_idm(blocks.values())
        c = block_data.COLOR_BID_MAP.get(colour)
        if c is not None:
            new_idm = random.choice(c)
            for l in blocks:
                if blocks[l] == old_idm:
                    blocks[l] = new_idm
    return list(blocks.items()), schematic.memid, tags


def interpret_schematic(
    interpreter, speaker, d
) -> List[Tuple[List[Block], Optional[str], List[Tuple[str, str]]]]:
    """Return a list of 3-tuples, each with values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """

    # FIXME! this is not compositional, and is not using full FILTERS handlers
    filters_d = d.get("filters", {})
    triples = filters_d.get("triples", [{"pred_text": "has_shape", "obj_text": "cube"}])
    shapes = get_properties_from_triples(triples, "has_shape")
    # AND this FIXME, not using selector properly
    repeat = filters_d.get("selector", {}).get("return_quantity", {}).get("random", "1")
    repeat = int(number_from_span(repeat))
    if any(shapes):
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
