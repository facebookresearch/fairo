"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.shared_data_struct.shared_data_structs import Time

"""This file contains utility functions for the CraftAssist agent"""
import copy

from math import sin, cos, pi
from typing import cast

from droidlet.shared_data_struct.base_util import *

TICKS_PER_MC_DAY = 24000
LOOK = Tuple[float, float]


class MCTime(Time):
    """Time in game"""

    def __init__(self, get_world_time):
        super().__init__()
        self.get_world_time = get_world_time

    def get_world_hour(self):
        """
        Returns:
            a fraction of a day.  0 is sunrise, .5 is sunset, 1.0 is next day
        """
        return self.get_world_time() / TICKS_PER_MC_DAY


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


def build_safe_diag_adjacent(bounds):
    """bounds is [mx, Mx, my, My, mz, Mz],
    if nothing satisfies, returns empty list"""

    def a(p):
        """Return the adjacent positions to p including diagonal adjaceny, within the bounds"""
        mx = max(bounds[0], p[0] - 1)
        my = max(bounds[2], p[1] - 1)
        mz = max(bounds[4], p[2] - 1)
        Mx = min(bounds[1] - 1, p[0] + 1)
        My = min(bounds[3] - 1, p[1] + 1)
        Mz = min(bounds[5] - 1, p[2] + 1)
        return [
            (x, y, z)
            for x in range(mx, Mx + 1)
            for y in range(my, My + 1)
            for z in range(mz, Mz + 1)
            if (x, y, z) != p
        ]

    return a


def cluster_areas(areas):
    """Cluster a list of areas so that intersected ones are unioned

    areas: list of tuple ((x, y, z), radius), each defines a cube
    where (x, y, z) is the center and radius is half the side length
    """

    def expand_xyzs(pos, radius):
        xmin, ymin, zmin = pos[0] - radius, pos[1] - radius, pos[2] - radius
        xmax, ymax, zmax = pos[0] + radius, pos[1] + radius, pos[2] + radius
        return xmin, xmax, ymin, ymax, zmin, zmax

    def is_intersecting(area1, area2):
        x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = expand_xyzs(area1[0], area1[1])
        x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = expand_xyzs(area2[0], area2[1])
        return (
            (x1_min <= x2_max and x1_max >= x2_min)
            and (y1_min <= y2_max and y1_max >= y2_min)
            and (z1_min <= z2_max and z1_max >= z2_min)
        )

    def merge_area(area1, area2):
        x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = expand_xyzs(area1[0], area1[1])
        x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = expand_xyzs(area2[0], area2[1])

        x_min, y_min, z_min = min(x1_min, x2_min), min(y1_min, y2_min), min(z1_min, z2_min)
        x_max, y_max, z_max = max(x1_max, x2_max), max(y1_max, y2_max), max(z1_max, z2_max)

        x, y, z = (x_min + x_max) // 2, (y_min + y_max) // 2, (z_min + z_max) // 2
        radius = max(
            (x_max - x_min + 1) // 2, max((y_max - y_min + 1) // 2, (z_max - z_min + 1) // 2)
        )
        return ((x, y, z), radius)

    unclustered_areas = copy.deepcopy(areas)
    clustered_areas = []

    while len(unclustered_areas) > 0:
        area = unclustered_areas[0]
        del unclustered_areas[0]
        finished = True
        idx = 0
        while idx < len(unclustered_areas) or not finished:
            if idx >= len(unclustered_areas):
                idx = 0
                finished = True
                continue
            if is_intersecting(area, unclustered_areas[idx]):
                area = merge_area(area, unclustered_areas[idx])
                finished = False
                del unclustered_areas[idx]
            else:
                idx += 1
        clustered_areas.append(area)

    return clustered_areas


def diag_adjacent(p):
    """Return the adjacent positions to p including diagonal adjaceny"""
    return [
        (x, y, z)
        for x in range(p[0] - 1, p[0] + 2)
        for y in range(p[1] - 1, p[1] + 2)
        for z in range(p[2] - 1, p[2] + 2)
        if (x, y, z) != p
    ]


def discrete_step_dir(agent):
    """Discretized unit vector in the direction of agent's yaw

    agent pos + discrete_step_dir = block in front of agent
    """
    yaw = agent.get_player().look.yaw
    x = round(-sin(yaw * pi / 180))
    z = round(cos(yaw * pi / 180))
    return np.array([x, 0, z], dtype="int32")


def fill_idmeta(agent, poss: List[XYZ]) -> List[Block]:
    """Add id_meta information to a a list of (xyz)s"""
    if len(poss) == 0:
        return []
    mx, my, mz = np.min(poss, axis=0)
    Mx, My, Mz = np.max(poss, axis=0)
    B = agent.get_blocks(mx, Mx, my, My, mz, Mz)
    idms = []
    for x, y, z in poss:
        idm = tuple(B[y - my, z - mz, x - mx])
        idms.append(cast(IDM, idm))
    return [(cast(XYZ, tuple(pos)), idm) for (pos, idm) in zip(poss, idms)]


def get_locs_from_entity(e):
    """Assumes input is either mob, memory, or tuple/list of coords
    outputs a tuple of coordinate tuples"""

    if hasattr(e, "pos"):
        if type(e.pos) is list or type(e.pos) is tuple or hasattr(e.pos, "dtype"):
            return (tuple(to_block_pos(e.pos)),)
        else:
            return tuple((tuple(to_block_pos(pos_to_np(e.pos))),))

    if str(type(e)).find("memory") > 0:
        if hasattr(e, "blocks"):
            return strip_idmeta(e.blocks)
        return None
    elif type(e) is tuple or type(e) is list:
        if len(e) > 0:
            if type(e[0]) is tuple:
                return e
            else:
                return tuple((e,))
    return None


# this should eventually be replaced with sql query
def most_common_idm(idms):
    """idms is a list of tuples [(id, m) ,.... (id', m')]"""
    counts = {}
    for idm in idms:
        if not counts.get(idm):
            counts[idm] = 1
        else:
            counts[idm] += 1
    return max(counts, key=counts.get)


def strip_idmeta(blockobj):
    """Return a list of (x, y, z) and drop the id_meta for blockobj"""
    if blockobj is not None:
        if type(blockobj) is dict:
            return list(pos for (pos, id_meta) in blockobj.items())
        else:
            return list(pos for (pos, id_meta) in blockobj)
    else:
        return None


def to_block_center(array):
    """Return the array centered at [0.5, 0.5, 0.5]"""
    return to_block_pos(array).astype("float") + [0.5, 0.5, 0.5]


def to_block_pos(array):
    """Convert array to block position"""
    return np.floor(array).astype("int32")
