"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import Tuple
import random
import Levenshtein

# TODO FILTERS!
def get_block_type(s, block_data_info, color_bid_map) -> Tuple:
    """string -> (id, meta)
    or  {"has_x": span} -> (id, meta)"""

    name_to_bid = block_data_info.get("name_to_bid", {})
    if type(s) is str:
        s_aug = s + " block"
        _, closest_match = min(
            [(name, id_meta) for (name, id_meta) in name_to_bid.items() if id_meta[0] < 256],
            key=lambda x: min(Levenshtein.distance(x[0], s), Levenshtein.distance(x[0], s_aug)),
        )
    else:
        if "has_colour" in s:
            c = color_bid_map.get(s["has_colour"])
            if c is not None:
                closest_match = random.choice(c)
        if "has_block_type" in s:
            _, closest_match = min(
                [(name, id_meta) for (name, id_meta) in name_to_bid.items() if id_meta[0] < 256],
                key=lambda x: min(
                    Levenshtein.distance(x[0], s), Levenshtein.distance(x[0], s_aug)
                ),
            )

    return closest_match
