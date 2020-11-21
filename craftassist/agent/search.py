"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import heapq
import logging
import numpy as np
import time

from block_data import PASSABLE_BLOCKS
from mc_util import adjacent, manhat_dist


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


class PriorityQueue:
    def __init__(self):
        self.q = []
        self.set = set()

    def push(self, x, prio):
        heapq.heappush(self.q, (prio, x))
        self.set.add(x)

    def pop(self):
        prio, x = heapq.heappop(self.q)
        self.set.remove(x)
        return prio, x

    def contains(self, x):
        return x in self.set

    def replace(self, x, newp):
        for i in range(len(self.q)):
            oldp, y = self.q[i]
            if x == y:
                self.q[i] = (newp, x)
                heapq.heapify(self.q)  # TODO: probably slow
                return
        raise ValueError("Not found: {}".format(x))

    def __len__(self):
        return len(self.q)


if __name__ == "__main__":
    X = np.ones((5, 5, 5), dtype="bool")
    X[3, :, :] = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    start = (3, 4, 0)
    goal = (3, 1, 0)
    print(astar(X, start, goal))
