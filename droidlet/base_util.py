"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import time
from collections import defaultdict, namedtuple
import binascii
import hashlib
import numpy as np
from word2number.w2n import word_to_num
from typing import Tuple, List, TypeVar
import uuid
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import PASSABLE_BLOCKS
from droidlet.shared_data_structs import PriorityQueue

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
            offsets = [np.round(np.asarray((offsets[i][0], offsets[i][2], 0))) for i in range(N)]
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


def astar(agent, target, approx=0, pos="agent"):
    """Find a path from the agent's pos to the target.

    Args:
    - agent: the Agent object
    - target: an absolute (x, y, z)
    - approx: proximity to target before search is complete (0 = exact)
    - pos: (optional) checks path from specified tuple

    Returns: a list of (x, y, z) positions from start to target
    """
    t_start = time.time()
    if type(pos) is str and pos == "agent":
        pos = agent.pos
    logging.debug("A* from {} -> {} ± {}".format(pos, target, approx))

    corners = np.array([pos, target]).astype("int32")
    mx, my, mz = corners.min(axis=0) - 10
    Mx, My, Mz = corners.max(axis=0) + 10
    my, My = max(my, 0), min(My, 255)
    blocks = agent.get_blocks(mx, Mx, my, My, mz, Mz)
    obstacles = np.isin(blocks[:, :, :, 0], PASSABLE_BLOCKS, invert=True)
    obstacles = obstacles[:-1, :, :] | obstacles[1:, :, :]  # check head and feet
    start, goal = (corners - [mx, my, mz])[:, [1, 2, 0]]
    path = _astar(obstacles, start, goal, approx)
    if path is not None:
        path = [(p[2] + mx, p[0] + my, p[1] + mz) for p in reversed(path)]

    t_elapsed = time.time() - t_start
    logging.debug("A* returned {}-len path in {}".format(len(path) if path else "None", t_elapsed))
    return path


def _astar(X, start, goal, approx=0):
    """Find a path through X from start to goal.

    Args:
    - X: a 3d array of obstacles, i.e. False -> passable, True -> not passable
    - start/goal: relative positions in X
    - approx: proximity to goal before search is complete (0 = exact)

    Returns: a list of relative positions, from start to goal
    """
    start = tuple(start)
    goal = tuple(goal)

    visited = set()
    came_from = {}
    q = PriorityQueue()
    q.push(start, manhat_dist(start, goal))
    G = np.full_like(X, np.iinfo(np.uint32).max, "uint32")
    G[start] = 0

    while len(q) > 0:
        _, p = q.pop()

        if manhat_dist(p, goal) <= approx:
            path = []
            while p in came_from:
                path.append(p)
                p = came_from[p]
            return [start] + list(reversed(path))

        visited.add(p)
        for a in adjacent(p):
            if (
                a in visited
                or a[0] < 0
                or a[0] >= X.shape[0]
                or a[1] < 0
                or a[1] >= X.shape[1]
                or a[2] < 0
                or a[2] >= X.shape[2]
                or X[a]
            ):
                continue

            g = G[p] + 1
            if g >= G[a]:
                continue
            came_from[a] = p
            G[a] = g
            f = g + manhat_dist(a, goal)
            if q.contains(a):
                q.replace(a, f)
            else:
                q.push(a, f)

    return None


def adjacent(p):
    """Return the positions adjacent to position p"""
    return (
        (p[0] + 1, p[1], p[2]),
        (p[0] - 1, p[1], p[2]),
        (p[0], p[1] + 1, p[2]),
        (p[0], p[1] - 1, p[2]),
        (p[0], p[1], p[2] + 1),
        (p[0], p[1], p[2] - 1),
    )


def depth_first_search(blocks_shape, pos, fn, adj_fn=adjacent):
    """Do depth-first search on array with blocks_shape starting
    from pos

    Calls fn(p) on each index `p` in DFS-order. If fn returns True,
    continue searching. If False, do not add adjacent blocks.

    Args:
    - blocks_shape: a tuple giving the shape of the blocks
    - pos: a relative position in blocks
    - fn: a function called on each position in DFS-order. Return
      True to continue searching from that node
    - adj_fn: a function (pos) -> list[pos], of adjacent positions

    Returns: visited, a bool array with blocks.shape
    """
    visited = np.zeros(blocks_shape, dtype="bool")
    q = [tuple(pos)]
    visited[tuple(pos)] = True
    i = 0
    while i < len(q):
        p = q.pop()
        if fn(p):
            for a in adj_fn(p):
                try:
                    if not visited[a]:
                        visited[a] = True
                        q.append(a)
                except IndexError:
                    pass
    return visited

# X = np.ones((5, 5, 5), dtype="bool")
#     X[3, :, :] = [
#         [0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [1, 1, 1, 1, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#     ]
#     start = (3, 4, 0)
#     goal = (3, 1, 0)
#     print(astar(X, start, goal))

def diag_adjacent(p):
    """Return the adjacent positions to p including diagonal adjaceny"""
    return [
        (x, y, z)
        for x in range(p[0] - 1, p[0] + 2)
        for y in range(p[1] - 1, p[1] + 2)
        for z in range(p[2] - 1, p[2] + 2)
        if (x, y, z) != p
    ]


def get_bounds(S):
    """
    S should be a list of tuples, where each tuple is a pair of
    (x, y, z) and ids;
    else a list of (x, y, z)
    """
    if len(S) == 0:
        return 0, 0, 0, 0, 0, 0
    if len(S[0]) == 3:
        T = [(l, (0, 0)) for l in S]
    else:
        T = S
    x, y, z = list(zip(*list(zip(*T))[0]))
    return min(x), max(x), min(y), max(y), min(z), max(z)