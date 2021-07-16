"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from collections import defaultdict, namedtuple
import binascii
import hashlib

import numpy
import numpy as np
from word2number.w2n import word_to_num
from typing import Tuple, List, TypeVar
import uuid

from droidlet.lowlevel.minecraft.shapes import get_bounds

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

def hash_user(username):
    """Encrypt username"""
    # uuid is used to generate a random number
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + username.encode()).hexdigest() + ":" + salt

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


MOBS_BY_ID = {
    50: "creeper",
    51: "skeleton",
    52: "spider",
    53: "giant",
    54: "zombie",
    55: "slime",
    56: "ghast",
    57: "pig zombie",
    58: "enderman",
    59: "cave spider",
    60: "silverfish",
    61: "blaze",
    62: "lava slime",
    63: "ender dragon",
    64: "wither boss",
    65: "bat",
    66: "witch",
    68: "guardian",
    90: "pig",
    91: "sheep",
    92: "cow",
    93: "chicken",
    94: "squid",
    95: "wolf",
    96: "mushroom cow",
    97: "snow man",
    98: "ozelot",
    99: "villager golem",
    100: "entity horse",
    101: "rabbit",
    120: "villager",
}


def blocks_list_to_npy(blocks, xyz=False):
    """Convert a list of blockid meta (x, y, z), (id, meta) to numpy"""
    xyzbm = np.array([(x, y, z, b, m) for ((x, y, z), (b, m)) in blocks])
    mx, my, mz = np.min(xyzbm[:, :3], axis=0)
    Mx, My, Mz = np.max(xyzbm[:, :3], axis=0)

    npy = np.zeros((My - my + 1, Mz - mz + 1, Mx - mx + 1, 2), dtype="int32")

    for x, y, z, b, m in xyzbm:
        npy[y - my, z - mz, x - mx] = (b, m)

    offsets = (my, mz, mx)

    if xyz:
        npy = np.swapaxes(np.swapaxes(npy, 1, 2), 0, 1)
        offsets = (mx, my, mz)

    return npy, offsets


def arrange(arrangement, schematic=None, shapeparams={}):
    """This function arranges an Optional schematic in a given arrangement
    and returns the offsets"""
    N = shapeparams.get("N", 7)
    extra_space = shapeparams.get("extra_space", 1)
    if schematic is None:
        bounds = [0, 1, 0, 1, 0, 1]
    else:
        bounds = get_bounds(schematic)

    if N <= 0:
        raise NotImplementedError(
            "TODO arrangement just based on extra space, need to specify number for now"
        )
    offsets = []
    if arrangement == "circle":
        orient = shapeparams.get("orient", "xy")
        encircled_object_radius = shapeparams.get("encircled_object_radius", 1)
        b = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        radius = max(((b + extra_space) * N) / (2 * np.pi), encircled_object_radius + b + 1)
        offsets = [
            (radius * np.cos(2 * s * np.pi / N), 0, radius * np.sin(2 * s * np.pi / N))
            for s in range(N)
        ]
        if orient == "yz":
            offsets = [np.round(np.asarray(0, offsets[i][0], offsets[i][2])) for i in range(N)]
        if orient == "xz":
            offsets = [
                np.round(np.asarray((offsets[i][0], offsets[i][2], 0))) for i in range(N)
            ]
    elif arrangement == "line":
        orient = shapeparams.get("orient")  # this is a vector here
        b = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        b += extra_space + 1
        offsets = [np.round(i * b * np.asarray(orient)) for i in range(N)]
    return offsets


def prepend_a_an(name):
    """Add a/an to a name"""
    if name[0] in ["a", "e", "i", "o", "u"]:
        return "an " + name
    else:
        return "a " + name


def to_block_pos(array):
    """Convert array to block position"""
    return np.floor(array).astype("int32")


def to_block_center(array):
    """Return the array centered at [0.5, 0.5, 0.5]"""
    return to_block_pos(array).astype("float") + [0.5, 0.5, 0.5]