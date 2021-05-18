"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from collections import defaultdict, namedtuple
import binascii
import hashlib
import logging
import numpy as np
import time
import traceback
from word2number.w2n import word_to_num
from typing import Tuple, List, TypeVar
import uuid

##FFS FIXME!!!! arrange utils properly, put things in one place
XYZ = Tuple[int, int, int]
# two points p0(x0, y0, z0), p1(x1, y1, z1) determine a 3d cube(point_at_target)
POINT_AT_TARGET = Tuple[int, int, int, int, int, int]
IDM = Tuple[int, int]
Block = Tuple[XYZ, IDM]
Hole = Tuple[List[XYZ], IDM]
T = TypeVar("T")  # generic type

#####FIXME!!!!!!  make all these dicts all through code
Pos = namedtuple("pos", ["x", "y", "z"])
Look = namedtuple("look", "yaw, pitch")
Player = namedtuple("Player", "entityId, name, pos, look")

TICKS_PER_SEC = 100
TICKS_PER_MINUTE = 60 * TICKS_PER_SEC
TICKS_PER_HOUR = 60 * TICKS_PER_MINUTE
TICKS_PER_DAY = 24 * TICKS_PER_HOUR


class ErrorWithResponse(Exception):
    def __init__(self, chat):
        self.chat = chat


class TimingWarn(object):
    """Context manager which logs a warning if elapsed time exceeds some threshold"""

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


def number_from_span(s):
    try:
        n = float(s)
    except:
        try:
            n = float(word_to_num(s))
        except:
            return
    return n


def check_username(hashed_username, username):
    """Compare the username with the hash to check if they
    are same"""
    user, salt = hashed_username.split(":")
    return user == hashlib.sha256(salt.encode() + username.encode()).hexdigest()


def get_bounds(locs):
    M = np.max(locs, axis=0)
    m = np.min(locs, axis=0)
    return m[0], M[0], m[1], M[1], m[2], M[2]


def group_by(items, key_fn):
    """Return a dict of {k: list[x]}, where key_fn(x) == k"""
    d = defaultdict(list)
    for x in items:
        d[key_fn(x)].append(x)
    return d


def hash_user(username):
    """Encrypt username"""
    # uuid is used to generate a random number
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + username.encode()).hexdigest() + ":" + salt


def euclid_dist(a, b):
    """Return euclidean distance between a and b"""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def manhat_dist(a, b):
    """Return mahattan ditance between a and b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def pos_to_np(pos):
    """Convert pos to numpy array"""
    if pos is None:
        return None
    return np.array((pos.x, pos.y, pos.z))


def shasum_file(path):
    """Retrn shasum of the file at path"""
    sha = hashlib.sha1()
    with open(path, "rb") as f:
        block = f.read(2 ** 16)
        while len(block) != 0:
            sha.update(block)
            block = f.read(2 ** 16)
    return binascii.hexlify(sha.digest())


# TODO make this just a dict, and change in memory and agent
# eg in object_looked_at and PlayerNode
def to_player_struct(pos, yaw, pitch, eid, name):
    if len(pos) == 2:
        pos = Pos(pos[0], 0.0, pos[1])
    else:
        pos = Pos(pos[0], pos[1], pos[2])
    look = Look(yaw, pitch)
    return Player(eid, name, pos, look)


def npy_to_blocks_list(npy, origin=(0, 0, 0)):
    """Convert a numpy array to block list ((x, y, z), (id, meta))"""
    blocks = []
    sy, sz, sx, _ = npy.shape
    for ry in range(sy):
        for rz in range(sz):
            for rx in range(sx):
                idm = tuple(npy[ry, rz, rx, :])
                if idm[0] == 0:
                    continue
                xyz = tuple(np.array([rx, ry, rz]) + origin)
                blocks.append((xyz, idm))
    return blocks


SPAWN_OBJECTS = {
    "elder guardian": 4,
    "wither skeleton": 5,
    "stray": 6,
    "husk": 23,
    "zombie villager": 27,
    "skeleton horse": 28,
    "zombie horse": 29,
    "donkey": 31,
    "mule": 32,
    "evoker": 34,
    "vex": 35,
    "vindicator": 36,
    "creeper": 50,
    "skeleton": 51,
    "spider": 52,
    "zombie": 54,
    "slime": 55,
    "ghast": 56,
    "zombie pigman": 57,
    "enderman": 58,
    "cave spider": 59,
    "silverfish": 60,
    "blaze": 61,
    "magma cube": 62,
    "bat": 65,
    "witch": 66,
    "endermite": 67,
    "guardian": 68,
    "shulker": 69,
    "pig": 90,
    "sheep": 91,
    "cow": 92,
    "chicken": 93,
    "squid": 94,
    "wolf": 95,
    "mooshroom": 96,
    "ocelot": 98,
    "horse": 100,
    "rabbit": 101,
    "polar bear": 102,
    "llama": 103,
    "parrot": 105,
    "villager": 120,
}


class NextDialogueStep(Exception):
    pass