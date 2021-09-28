"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file contains utility functions for the CraftAssist agent"""
import copy
from math import sin, cos, pi
from typing import cast
from droidlet.base_util import *
from droidlet.shared_data_structs import Time

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




def strip_idmeta(blockobj):
    """Return a list of (x, y, z) and drop the id_meta for blockobj"""
    if blockobj is not None:
        if type(blockobj) is dict:
            return list(pos for (pos, id_meta) in blockobj.items())
        else:
            return list(pos for (pos, id_meta) in blockobj)
    else:
        return None


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