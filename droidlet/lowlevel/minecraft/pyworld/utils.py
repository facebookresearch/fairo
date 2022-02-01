"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from collections import namedtuple
from unittest.mock import Mock

import numpy as np

Player = namedtuple("Player", "entityId, name, pos, look, mainHand")
Mob = namedtuple("Mob", "entityId, mobType, pos, look")
Pos = namedtuple("Pos", "x, y, z")
Look = namedtuple("Look", "yaw, pitch")
Item = namedtuple("Item", "id, meta")


class PickleMock(Mock):
    """Mocks cannot be pickled. This Mock class can."""

    def __reduce__(self):
        return (Mock, ())


def to_relative_pos(block_list):
    """Convert absolute block positions to their relative positions

    Find the "origin", i.e. the minimum (x, y, z), and subtract this from all
    block positions.

    Args:
    - block_list: a list of ((x,y,z), (id, meta))

    Returns:
    - a block list where positions are shifted by `origin`
    - `origin`, the (x, y, z) offset by which the positions were shifted
    """
    try:
        locs, idms = zip(*block_list)
    except ValueError:
        raise ValueError("to_relative_pos invalid input: {}".format(block_list))

    locs = np.array([loc for (loc, idm) in block_list])
    origin = np.min(locs, axis=0)
    locs -= origin
    S = [(tuple(loc), idm) for (loc, idm) in zip(locs, idms)]
    if type(block_list) is not list:
        S = tuple(S)
    if type(block_list) is frozenset:
        S = frozenset(S)
    return S, origin
