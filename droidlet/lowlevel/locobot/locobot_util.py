"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys
from collections import defaultdict
import binascii
import hashlib
import logging
import numpy as np
import time
import traceback
import uuid
from . import rotation
from droidlet.shared_data_struct.base_util import XYZ, euclid_dist
from typing import Tuple, List, TypeVar, Sequence


class Pos:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def pos_to_np(pos):
    """Convert pos to numpy array."""
    if pos is None:
        return None
    return np.array((pos.x, pos.y, pos.z))


def check_username(hashed_username, username):
    """Compare the username with the hash to check if they are same."""
    user, salt = hashed_username.split(":")
    return user == hashlib.sha256(salt.encode() + username.encode()).hexdigest()


def base_distance(x, y):
    if y[0] is None or y[2] is None:
        return 10000000.0
    else:
        return (x[0] - y[0]) ** 2 + (x[1] - y[2]) ** 2


#######################################!!!!!!!!!!
# TODO move this to "reasoning"
# THIS NEEDS TO BE REWRITTEN TO MATCH LOCOBOT
def object_looked_at(
        agent,
        candidates: Sequence[Tuple[XYZ, T]],
        player_struct,
        limit=1,
        max_distance=30,
        loose=False,
) -> List[Tuple[XYZ, T]]:
    """Return the object that `player` is looking at.

    Args:
    - agent: agent object, for API access
    - candidates: list of (centroid, object) tuples
    - player_struct: player struct whose POV to use for calculation
         has a .pos attribute, which is an (x, y, z) tuple giving the head position
         and a .pitch attribute, which is a float
         and a .yaw attribute, which is a float
    - limit: 'ALL' or int; max candidates to return
    - loose:  if True, don't filter candaidates behind agent

    Returns: a list of (xyz, mem) tuples, max length `limit`
    """
    if len(candidates) == 0:
        return []
    # FIXME !!!! formalize this:
    # should not even end up here if true, handle above.
    if player_struct.pos.x is None:
        # speaker is "disembodied", return object closest to agent
        # todo closest to agent line of sight?
        candidates.sort(key=lambda c: base_distance(agent.pos, c[0]))
        # limit returns of things too far away
        candidates = [c for c in candidates if base_distance(agent.pos, c[0]) < max_distance]
        return [(p, o) for (p, o, r) in candidates[:limit]]

    pos = np.array(player_struct.pos)
    yaw, pitch = player_struct.look.yaw, player_struct.look.pitch

    # append to each candidate its relative position to player, rotated to
    # player-centric coordinates
    candidates_ = [(p, obj, rotation.transform(p - pos, yaw, pitch)) for (p, obj) in candidates]
    FRONT = rotation.DIRECTIONS["FRONT"]
    LEFT = rotation.DIRECTIONS["LEFT"]
    UP = rotation.DIRECTIONS["UP"]

    # reject objects behind player or not in cone of sight (but always include
    # an object if it's directly looked at)
    xsect = tuple(capped_line_of_sight(agent, player_struct, 25))
    if not loose:
        candidates_ = [
            (p, o, r)
            for (p, o, r) in candidates_
            if xsect in getattr(o, "blocks", {})
               or r @ FRONT > ((r @ LEFT) ** 2 + (r @ UP) ** 2) ** 0.5
        ]

    # if looking directly at an object, sort by proximity to look intersection
    if euclid_dist(pos, xsect) <= 25:
        candidates_.sort(key=lambda c: euclid_dist(c[0], xsect))
    else:
        # otherwise, sort by closest to look vector
        candidates_.sort(key=lambda c: ((c[2] @ LEFT) ** 2 + (c[2] @ UP) ** 2) ** 0.5)
    # linit returns of things too far away
    candidates_ = [c for c in candidates_ if euclid_dist(pos, c[0]) < max_distance]
    # limit number of returns
    if limit == "ALL":
        limit = len(candidates_)
    return [(p, o) for (p, o, r) in candidates_[:limit]]


#######################################!!!!!!!!!!
# THIS NEEDS TO BE REWRITTEN TO MATCH LOCOBOT
def capped_line_of_sight(agent, player_struct, cap=20):
    """Return the block directly in the entity's line of sight, or a point in
    the distance."""
    xsect = agent.get_player_line_of_sight(player_struct)
    if xsect is not None and euclid_dist(pos_to_np(xsect), pos_to_np(player_struct.pos)) <= cap:
        return pos_to_np(xsect)

    # default to cap blocks in front of entity
    vec = rotation.look_vec(player_struct.look.yaw, player_struct.look.pitch)
    return cap * np.array(vec) + pos_to_np(player_struct.pos)
