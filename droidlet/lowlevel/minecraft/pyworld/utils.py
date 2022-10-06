"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from collections import namedtuple
from unittest.mock import Mock

import numpy as np

FLAT_GROUND_DEPTH = 8

BEDROCK = (7, 0)
DIRT = (3, 0)
GRASS = (2, 0)
AIR = (0, 0)


class PickleMock(Mock):
    """Mocks cannot be pickled. This Mock class can."""

    def __reduce__(self):
        return (Mock, ())


def to_relative_pos(block_list):
    """Convert absolute block positions to their relative positions

    Find the "origin", i.e. the minimum (x, y, z), and subtract this from all
    block positions.

    Args:
    - block_list: a list of ((x,y,z), (id, meta))

    Returns:
    - a block list where positions are shifted by `origin`
    - `origin`, the (x, y, z) offset by which the positions were shifted
    """
    try:
        locs, idms = zip(*block_list)
    except ValueError:
        raise ValueError("to_relative_pos invalid input: {}".format(block_list))

    locs = np.array([loc for (loc, idm) in block_list])
    origin = np.min(locs, axis=0)
    locs -= origin
    S = [(tuple(loc), idm) for (loc, idm) in zip(locs, idms)]
    if type(block_list) is not list:
        S = tuple(S)
    if type(block_list) is frozenset:
        S = frozenset(S)
    return S, origin


def shift_coords(p, shift):
    if hasattr(p, "x"):
        return Pos(p.x + shift[0], p.y + shift[1], p.z + shift[2])
    q = np.add(p, shift)
    if type(p) is tuple:
        q = tuple(q)
    if type(p) is list:
        q = list(q)
    return q


def build_coord_shifts(coord_shift):
    def to_npy_coords(p):
        dx = -coord_shift[0]
        dy = -coord_shift[1]
        dz = -coord_shift[2]
        return shift_coords(p, (dx, dy, dz))

    def from_npy_coords(p):
        return shift_coords(p, coord_shift)

    return to_npy_coords, from_npy_coords


def flat_ground_generator_with_grass(world):
    flat_ground_generator(world, grass=True)


def flat_ground_generator(world, grass=False, ground_depth=FLAT_GROUND_DEPTH):
    r = world.to_npy_coords((0, 62, 0))[1]
    # r = world.sl // 2
    # fill world with air:
    world.blocks[:] = 0
    # fill everything below r with bedrock
    world.blocks[:, 0:r, :, 0] = BEDROCK[0]
    # dirt
    world.blocks[:, r - ground_depth : r, :, 0] = DIRT[0]
    if grass:
        world.blocks[:, r, :, 0] = GRASS[0]
    else:
        world.blocks[:, r, :, 0] = DIRT[0]


def build_ground(world):
    if hasattr(world.opts, "avg_ground_height"):
        avg_ground_height = world.opts.avg_ground_height
    else:
        avg_ground_height = 6.0
    if hasattr(world.opts, "hill_scale"):
        hill_scale = world.opts.hill_scale
    else:
        hill_scale = 5.0
    p = hill_scale * np.random.randn(6)
    g = np.mgrid[0 : world.sl, 0 : world.sl].astype("float32") / world.sl
    ground_height = (
        p[0] * np.sin(g[0])
        + p[1] * np.cos(g[0])
        + p[2] * np.cos(g[0]) * np.sin(g[0])
        + p[3] * np.sin(g[1])
        + p[4] * np.cos(g[1])
        + p[5] * np.cos(g[1]) * np.sin(g[1])
    )
    ground_height = ground_height - ground_height.mean() + avg_ground_height
    for i in range(world.sl):
        for j in range(world.sl):
            height = min(31, max(0, int(ground_height[i, j])))
            for k in range(int(height)):
                world.blocks[i, k, j] = DIRT

    # FIXME this is broken
    if hasattr(world.opts, "ground_block_probs"):
        ground_blocks = np.transpose(np.nonzero(world.blocks[:, :, :, 0] == 3))
        num_ground_blocks = len(ground_blocks)
        for idm, val in world.opts.ground_block_probs:
            if idm != DIRT:
                num = np.random.rand() * val * 2 * num_ground_blocks
                for i in range(num):
                    j = np.random.randint(num_ground_blocks)
                    world.blocks[
                        ground_blocks[j][0], ground_blocks[j][1], ground_blocks[j][2], :
                    ] = idm


# THIS IS A DUPLICATE of the method in small_scenes_with_shapes.  dedup!
def make_pose(SL, H, loc=None, pitchyaw=None, height_map=None):
    """
    make a random pose for player or mob.
    if loc or pitchyaw is specified, use those
    if height_map is specified, finds a point close to the loc
        1 block higher than the height_map, but less than ENTITY_HEIGHT from
        H
    TODO option to input object locations and pick pitchyaw to look at one
    """
    ENTITY_HEIGHT = 2
    if loc is None:
        x, y, z = np.random.randint((SL / 4 - 1, H / 4 - 1, SL / 4 - 1)) + SL / 2
    else:
        x, y, z = loc
    if pitchyaw is None:
        pitch = np.random.uniform(-np.pi / 2, np.pi / 2)
        yaw = np.random.uniform(-np.pi, np.pi)
    else:
        pitch, yaw = pitchyaw
    # put the entity above the current height map.  this will break if
    # there is a big flat slab covering the entire space high, FIXME
    if height_map is not None:
        okh = np.array(np.nonzero(height_map < H - ENTITY_HEIGHT))
        if okh.shape[1] == 0:
            raise Exception(
                "no space for entities, height map goes up to H-ENTITY_HEIGHT everywhere"
            )
        d = np.linalg.norm((okh - np.array((x, z)).reshape(2, 1)), 2, 0)
        minidx = np.argmin(d)
        x = int(okh[0, minidx])
        z = int(okh[1, minidx])
        y = int(height_map[x, z] + 1)
    return x, y, z, pitch, yaw
