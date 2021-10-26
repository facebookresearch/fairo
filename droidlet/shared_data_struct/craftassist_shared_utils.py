import logging
import time
import numpy as np
from collections import namedtuple

from droidlet.base_util import adjacent, get_bounds, manhat_dist
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import PASSABLE_BLOCKS
from droidlet.shared_data_structs import PriorityQueue

CraftAssistPerceptionData = namedtuple("perception_data",
                                       ["holes", "mobs", "agent_pickable_items",
                                        "agent_attributes", "other_player_list", "changed_block_attributes",
                                        "in_perceive_area", "near_agent", "labeled_blocks"],
                                       defaults=[None, [], {}, None, [], {}, {}, {}, {}])

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