"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import random
import re
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Union, Optional

# TODO with subinterpret
from droidlet.memory.craftassist.mc_memory_nodes import SchematicNode
from droidlet.interpreter.craftassist import size_words
from .block_handler import get_block_type
from droidlet.base_util import number_from_span, Block
from droidlet.shared_data_structs import ErrorWithResponse
from word2number.w2n import word_to_num
from droidlet.interpreter.craftassist.word_maps import SPECIAL_SHAPES_CANONICALIZE
from droidlet.interpreter import interpret_where_backoff, maybe_apply_selector

############################################################################
# Plan: store virtual memories / generative models, and search over those
# Until then FILTERs in schematics are not fully compositional nor expressive
############################################################################


# this should eventually be replaced with db query
def most_common_idm(idms):
    """idms is a list of tuples [(id, m) ,.... (id', m')]"""
    counts = {}
    for idm in idms:
        if not counts.get(idm):
            counts[idm] = 1
        else:
            counts[idm] += 1
    return max(counts, key=counts.get)


def get_flattened_clauses(filters_d, default={}):
    where_clause = filters_d.get("where_clause", {})
    # FIXME!  maybe wait till virtual memories:  separate out the flattened clauses and others
    return [
        c for c in where_clause.get("AND", [default]) if ("value_left" in c or "pred_text" in c)
    ]


def get_triples_from_flattened_clauses(flattened_clauses):
    triples = []
    for t in flattened_clauses:
        key = t.get("pred_text", "")
        if key.startswith("has_"):
            val = t.get("obj_text", "")
            if val and type(val) is str:
                triples.append((key, val))
    return triples


# this does not allow a list of props for a given key, TODO?
def get_properties_from_clauses(clause_list, predicates):
    props = {}
    for clause in clause_list:
        if clause.get("pred_text") in predicates:
            # it wouldn't be too hard to recurse here, TODO?
            if type(clause.get("obj_text", {})) is not dict:
                props[clause["pred_text"]] = clause["obj_text"]
        if clause.get("value_left") in predicates:
            if clause.get("comparison_type", "EQUAL") == "EQUAL":
                if clause.get("value_right", {}) is not dict:
                    props[clause["value_left"]] = clause["value_right"]
    return props


def get_attrs_from_where(where_clauses, interpreter, block_data_info, color_bid_map):
    numeric_props = [
        "has_thickness",
        "has_radius",
        "has_depth",
        "has_width",
        "has_height",
        "has_length",
        "has_slope",
        "has_distance",
        "has_base",
    ]
    numeric_keys = get_properties_from_clauses(where_clauses, numeric_props)
    attrs = {key[4:]: word_to_num(val[0]) for key, val in numeric_keys.items()}

    text_props = ["has_orientation", "has_size", "has_block_type", "has_colour"]
    text_keys = get_properties_from_clauses(where_clauses, text_props)

    if text_keys.get("has_orientation"):
        attrs["orient"] = text_keys["has_orientation"]

    if text_keys.get("has_size"):
        attrs["size"] = interpret_size(interpreter, text_keys["has_size"])

    if text_keys.get("has_block_type"):
        block_type = get_block_type(
            text_keys["has_block_type"],
            block_data_info=block_data_info,
            color_bid_map=color_bid_map,
        )
        attrs["bid"] = block_type
    elif text_keys.get("has_colour"):
        c = color_bid_map.get(text_keys["has_colour"])
        if c is not None:
            attrs["bid"] = random.choice(c)
    else:
        pass

    return attrs


# FIXME merge with shape_schematic
# FIXME we should be able to do fancy stuff here, like fill the x with (copies of) schematic y
def interpret_fill_schematic(
    interpreter, speaker, d, hole_locs, hole_idm, block_data_info, color_bid_map
) -> Tuple[List[Block], List[Tuple[str, str]]]:
    """Return a tuple of 2 values:
    - the schematic blocks, list[(xyz, idm)]
    - a list of (pred, val) tags

    the "hole" input is a list of xyz coordinates giving a "mold" to be filled.
    """

    filters_d = d.get("filters", {})
    flattened_clauses = get_flattened_clauses(filters_d)
    attrs = get_attrs_from_where(flattened_clauses, interpreter, block_data_info, color_bid_map)

    h = attrs.get("height") or attrs.get("depth") or attrs.get("thickness")
    bid = attrs.get("bid") or hole_idm or (1, 0)
    origin = np.min(hole_locs, axis=0)
    ymin = origin[1]
    if h:
        blocks_list = [((x, y, z), bid) for (x, y, z) in hole_locs if y - ymin < h]
    else:
        blocks_list = [((x, y, z), bid) for (x, y, z) in hole_locs]
    tags = []
    triples = get_triples_from_flattened_clauses(flattened_clauses)
    return blocks_list, triples


def interpret_shape_schematic(
    interpreter,
    speaker,
    d,
    block_data_info,
    color_bid_map,
    special_shape_function,
    shapename=None,
    is_dig=False,
) -> Tuple[List[Block], List[Tuple[str, str]]]:
    """Return a tuple of 2 values:
    - the schematic blocks, list[(xyz, idm)]
    - a list of (pred, val) tags
    """
    # FIXME this is not compositional, and does not properly use FILTERS
    filters_d = d.get("filters", {})
    cube_triple = [{"pred_text": "has_shape", "obj_text": "cube"}]
    flattened_clauses = get_flattened_clauses(filters_d, default=cube_triple)
    attrs = get_attrs_from_where(flattened_clauses, interpreter, block_data_info, color_bid_map)

    shape = ""
    if shapename is not None:
        shape = shapename
    else:
        # For sentences like "Stack" and "Place" that have the shapename in dict
        shape = get_properties_from_clauses(flattened_clauses, ["has_shape"]).get(
            "has_shape", "cube"
        )

    triples = get_triples_from_flattened_clauses(flattened_clauses)
    schematic = special_shape_function[shape.upper()](**attrs)
    return schematic, triples


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
        return size_words.size_str_to_int(text)


def interpret_named_schematic(
    interpreter, speaker, d, block_data_info, color_bid_map, special_shape_function
) -> Tuple[List[Block], Optional[str], List[Tuple[str, str]]]:
    """Return a tuple of 3 values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """
    # FIXME! this is not compositional, and is not using full FILTERS handlers
    filters_d = d.get("filters", {})
    flattened_clauses = get_flattened_clauses(filters_d)
    name = get_properties_from_clauses(flattened_clauses, ["has_name"]).get("has_name", "")
    if not name:
        raise ErrorWithResponse("I don't know what you want me to build.")
    stemmed_name = name.strip("s")  # why aren't we using stemmer anymore?
    shapename = SPECIAL_SHAPES_CANONICALIZE.get(name) or SPECIAL_SHAPES_CANONICALIZE.get(
        stemmed_name
    )
    if shapename:
        shape_blocks, tags = interpret_shape_schematic(
            interpreter,
            speaker,
            d,
            block_data_info,
            color_bid_map,
            special_shape_function,
            shapename=shapename,
        )
        return shape_blocks, None, tags

    schematic = interpreter.memory.get_schematic_by_name(name)
    if schematic is None:
        schematic = interpreter.memory.get_schematic_by_name(stemmed_name)
        if schematic is None:
            raise ErrorWithResponse("I don't know what you want me to build.")
    triples = [(p, v) for (_, p, v) in interpreter.memory.get_triples(subj=schematic.memid)]
    blocks = schematic.blocks
    # TODO generalize to more general block properties
    # Longer term: remove and put a call to the modify model here
    colour = get_properties_from_clauses(flattened_clauses, ["has_colour"]).get("has_colour", "")
    if colour:
        old_idm = most_common_idm(blocks.values())
        c = color_bid_map.get(colour)
        if c is not None:
            new_idm = random.choice(c)
            for l in blocks:
                if blocks[l] == old_idm:
                    blocks[l] = new_idm
    return list(blocks.items()), schematic.memid, triples


def interpret_schematic(
    interpreter, speaker, d, block_data_info, color_bid_map, special_shape_function
) -> List[Tuple[List[Block], Optional[str], List[Tuple[str, str]]]]:
    """Return a list of 3-tuples, each with values:
    - the schematic blocks, list[(xyz, idm)]
    - a SchematicNode memid, or None
    - a list of (pred, val) tags
    """

    # FIXME! this is not compositional, and is not using full FILTERS handlers
    filters_d = d.get("filters", {})
    flattened_clauses = get_flattened_clauses(filters_d)
    shape = get_properties_from_clauses(flattened_clauses, ["has_shape"]).get("has_shape")
    # AND this FIXME, not using selector properly
    repeat = filters_d.get("selector", {}).get("ordinal", "1")
    repeat = int(number_from_span(repeat))
    if shape:
        blocks, tags = interpret_shape_schematic(
            interpreter, speaker, d, block_data_info, color_bid_map, special_shape_function
        )
        return [(blocks, None, tags)] * repeat
    else:
        return [
            interpret_named_schematic(
                interpreter, speaker, d, block_data_info, color_bid_map, special_shape_function
            )
        ] * repeat


def get_repeat_dir(d):
    if "repeat" in d:
        direction_name = d.get("repeat", {}).get("repeat_dir", "FRONT")
    elif "schematic" in d:
        direction_name = d["schematic"].get("repeat", {}).get("repeat_dir", "FRONT")
    else:
        direction_name = None
    return direction_name


def interpret_mob_schematic(interpreter, speaker, filters_d):
    spawn_clause = {"pred_text": "has_tag", "obj_text": "_spawn"}
    where = filters_d.get("where_clause", {"AND": [spawn_clause]})
    if where.get("AND"):
        where["AND"].append(spawn_clause)
    else:
        new_where = {"AND": [spawn_clause, deepcopy(where)]}
        where = new_where

    # HACK for nsp/data weirdness: for now don't allow
    # 'same': 'DISALLOWED' in Selector so could not e.g.
    # "spawn three different kinds of mobs".  we don't have examples
    # like that yet anyway ...

    if filters_d.get("selector", {}):
        if filters_d["selector"].get("same", "ALLOWED") == "DISALLOWED":
            filters_d["selector"]["same"] = "ALLOWED"

    # FIXME! we don't need to recopy this here, do more composably
    W = interpret_where_backoff(interpreter, speaker, where, memory_type="Schematic")
    F = maybe_apply_selector(interpreter, speaker, filters_d, W)
    schematic_memids, _ = F()
    object_idms = [
        list(SchematicNode(interpreter.memory, m).blocks.values())[0] for m in schematic_memids
    ]
    if not object_idms:
        raise ErrorWithResponse("I don't know how to spawn that")
    return object_idms
