"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from collections import defaultdict, namedtuple
import binascii
import hashlib
import numpy as np
from word2number.w2n import word_to_num
from typing import Tuple, List, TypeVar
import uuid

from droidlet.perception.craftassist.shapes import DEFAULT_IDM, rectanguloid

"""FIXME!!!! arrange utils properly, put things in one place"""
XYZ = Tuple[int, int, int]
# two points p0(x0, y0, z0), p1(x1, y1, z1) determine a 3d cube(point_at_target)
POINT_AT_TARGET = Tuple[int, int, int, int, int, int]
IDM = Tuple[int, int]
Block = Tuple[XYZ, IDM]
Hole = Tuple[List[XYZ], IDM]
T = TypeVar("T")  # generic type

"""FIXME!!!!!!  make all these dicts all through code"""
Pos = namedtuple("pos", ["x", "y", "z"])
Look = namedtuple("look", "yaw, pitch")
Player = namedtuple("Player", "entityId, name, pos, look")

TICKS_PER_SEC = 100
TICKS_PER_MINUTE = 60 * TICKS_PER_SEC
TICKS_PER_HOUR = 60 * TICKS_PER_MINUTE
TICKS_PER_DAY = 24 * TICKS_PER_HOUR


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
    """Return manhattan distance between a and b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def pos_to_np(pos):
    """Convert pos to numpy array"""
    if pos is None:
        return None
    return np.array((pos.x, pos.y, pos.z))


def shasum_file(path):
    """Return shasum of the file at a given path"""
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


def cube(size=3, bid=DEFAULT_IDM, labelme=False, **kwargs):
    if type(size) not in (tuple, list):
        size = (size, size, size)
    return rectanguloid(size=size, bid=bid, labelme=labelme)