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

from typing import Tuple, List, TypeVar, Sequence

XYZ = Tuple[int, int, int]
# two points p0(x0, y0, z0), p1(x1, y1, z1) determine a 3d cube(point_at_target)
POINT_AT_TARGET = Tuple[int, int, int, int, int, int]
IDM = Tuple[int, int]
Block = Tuple[XYZ, IDM]
Hole = Tuple[List[XYZ], IDM]
T = TypeVar("T")  # generic type


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


def euclid_dist(a, b):
    """Return euclidean distance between a and b."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def manhat_dist(a, b):
    """Return mahattan ditance between a and b."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def group_by(items, key_fn):
    """Return a dict of {k: list[x]}, where key_fn(x) == k."""
    d = defaultdict(list)
    for x in items:
        d[key_fn(x)].append(x)
    return d


def hash_user(username):
    """Encrypt username."""
    # uuid is used to generate a random number
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + username.encode()).hexdigest() + ":" + salt


def check_username(hashed_username, username):
    """Compare the username with the hash to check if they are same."""
    user, salt = hashed_username.split(":")
    return user == hashlib.sha256(salt.encode() + username.encode()).hexdigest()


def shasum_file(path):
    """Retrn shasum of the file at path."""
    sha = hashlib.sha1()
    with open(path, "rb") as f:
        block = f.read(2 ** 16)
        while len(block) != 0:
            sha.update(block)
            block = f.read(2 ** 16)
    return binascii.hexlify(sha.digest())


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


class TimingWarn(object):
    """Context manager which logs a warning if elapsed time exceeds some
    threshold."""

    def __init__(self, max_time: float):
        self.max_time = max_time

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= self.max_time:
            logging.warn(
                "Timing exceeded threshold: {}".format(self.elapsed_time)
                + "\n"
                + "".join(traceback.format_stack(limit=2))
            )
