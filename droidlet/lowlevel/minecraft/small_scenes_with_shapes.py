"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import random
from droidlet.lowlevel.minecraft.iglu_util import IGLU_BLOCK_MAP
from droidlet.lowlevel.minecraft.shape_util import (
    SHAPE_NAMES,
    SHAPE_FNS,
    SHAPE_OPTION_FUNCTION_MAP,
)

SL = 17
GROUND_DEPTH = 5
H = 13

HOLE_NAMES = ["RECTANGULOID", "ELLIPSOID", "SPHERE"]


def bid(nowhite=True):
    if nowhite:
        return (35, np.random.randint(15) + 1)
    else:
        return (35, np.random.randint(16))


def red():
    return (35, 14)


def white():
    return (35, 0)


# ignoring blocks for now
def avatar_pos(args, blocks):
    return (1, args.GROUND_DEPTH, 1)


def avatar_look(args, blocks):
    return (0.0, 0.0)


def agent_pos(args, blocks):
    return (3, args.GROUND_DEPTH, 0)


def agent_look(args, blocks):
    return (0.0, 0.0)


def build_base_world(sl, h, g, fence=False):
    W = {}
    for i in range(sl):
        for j in range(g):
            for k in range(sl):
                if (i == 0 or i == sl - 1 or k == 0 or k == sl - 1) and j == g - 1 and fence:
                    idm = red()
                else:
                    idm = white()
                W[(i, j, k)] = idm
    return W


def shift(blocks, s):
    for i in range(len(blocks)):
        b = blocks[i]
        if len(b) == 2:
            l, idm = b
            blocks[i] = ((l[0] + s[0], l[1] + s[1], l[2] + s[2]), idm)
        else:
            assert len(b) == 3
            blocks[i] = (b[0] + s[0], b[1] + s[1], b[2] + s[2])
    return blocks


def in_box_builder(mx, my, mz, Mx, My, Mz):
    def f(l):
        lb = l[0] >= mx and l[1] >= my and l[2] >= mz
        ub = l[0] < Mx and l[1] < My and l[2] < Mz
        return lb and ub

    return f


def record_shape(S, in_box, offsets, blocks, inst_seg, occupied_by_shapes):
    for l, idm in S:
        ln = np.add(l, offsets)
        if in_box(ln):
            ln = tuple(ln.tolist())
            if not occupied_by_shapes.get(ln):
                blocks[ln] = idm
                inst_seg.append(ln)
                occupied_by_shapes[ln] = True


def collect_scene(blocks, inst_segs, args):
    J = {}
    # FIXME not using the avatar and agent position in cuberite...
    J["avatarInfo"] = {"pos": avatar_pos(args, blocks), "look": avatar_look(args, blocks)}
    J["agentInfo"] = {"pos": agent_pos(args, blocks), "look": agent_look(args, blocks)}
    J["inst_seg_tags"] = inst_segs
    mapped_blocks = [(l[0], l[1], l[2], IGLU_BLOCK_MAP[idm]) for l, idm in blocks]
    J["blocks"] = mapped_blocks

    o = (0, args.cuberite_y_offset, 0)
    blocks = shift(blocks, o)
    J["schematic_for_cuberite"] = [
        {"x": l[0], "y": l[1], "z": l[2], "id": idm[0], "meta": idm[1]} for l, idm in blocks
    ]
    J["offset"] = (args.cuberite_x_offset, args.cuberite_y_offset, args.cuberite_z_offset)
    return J


def build_shape_scene(args):
    """
    Build a scene using basic shapes,
    outputs a json dict with fields
    "avatarInfo" = {"pos": (x,y,z), "look": (yaw, pitch)}
    "agentInfo" = {"pos": (x,y,z), "look": (yaw, pitch)}
    "blocks" = [(x,y,z,bid) ... (x,y,z,bid)]
    "schematic_for_cuberite" = [{"x": x, "y":y, "z":z, "id":blockid, "meta": meta} ...]
    where bid is the output of the BLOCK_MAP applied to a minecraft blockid, meta pair.
    """
    fence = getattr(args, "fence", False)
    blocks = build_base_world(args.SL, args.H, args.GROUND_DEPTH, fence=fence)
    num_shapes = np.random.randint(0, args.MAX_NUM_SHAPES + 1)
    occupied_by_shapes = {}
    inst_segs = []
    for t in range(num_shapes):
        shape = random.choice(SHAPE_NAMES)
        opts = SHAPE_OPTION_FUNCTION_MAP[shape]()
        opts["bid"] = bid()
        S = SHAPE_FNS[shape](**opts)
        m = np.round(np.mean([l for l, idm in S], axis=0)).astype("int32")
        offsets = np.random.randint((args.SL, args.H, args.SL)) - m
        inst_seg = []
        in_box = in_box_builder(0, 0, 0, args.SL, args.H, args.SL)
        record_shape(S, in_box, offsets, blocks, inst_seg, occupied_by_shapes)
        inst_segs.append({"tags": [shape], "locs": inst_seg})

    if args.MAX_NUM_GROUND_HOLES == 0:
        num_holes = 0
    else:
        num_holes = np.random.randint(0, args.MAX_NUM_GROUND_HOLES)
    # TODO merge contiguous holes
    ML = args.SL
    mL = 0
    if args.fence:
        ML -= 1
        mL = 1
    for t in range(num_holes):
        shape = random.choice(HOLE_NAMES)
        opts = SHAPE_OPTION_FUNCTION_MAP[shape]()
        S = SHAPE_FNS[shape](**opts)
        S = [(l, (0, 0)) for l, idm in S]
        m = np.round(np.mean([l for l, idm in S], axis=0)).astype("int32")
        miny = min([l[1] for l, idm in S])
        maxy = max([l[1] for l, idm in S])
        offsets = np.random.randint((args.SL, args.H, args.SL))
        offsets[0] -= m[0]
        offsets[2] -= m[2]
        # offset miny to GROUND_DEPTH - radius of shape
        offsets[1] = args.GROUND_DEPTH - maxy // 2 - 1
        inst_seg = []
        in_box = in_box_builder(mL, 0, mL, ML, args.GROUND_DEPTH, ML)
        record_shape(S, in_box, offsets, blocks, inst_seg, occupied_by_shapes)
        inst_segs.append({"tags": ["hole"], "locs": inst_seg})
    J = {}
    # not shifting y for gridworld
    o = (args.cuberite_x_offset, 0, args.cuberite_z_offset)
    blocks = [(l, idm) for l, idm in blocks.items()]
    blocks = shift(blocks, o)
    for i in inst_segs:
        i["locs"] = shift(i["locs"], o)

    return collect_scene(blocks, inst_segs, args)


def build_extra_simple_shape_scene(args):
    """
    Build a scene with a sphere and a cube, non-overlapping.
    outputs a json dict with fields
    "avatarInfo" = {"pos": (x,y,z), "look": (yaw, pitch)}
    "agentInfo" = {"pos": (x,y,z), "look": (yaw, pitch)}
    "blocks" = [(x,y,z,bid) ... (x,y,z,bid)]
    "schematic_for_cuberite" = [{"x": x, "y":y, "z":z, "id":blockid, "meta": meta} ...]
    where bid is the output of the BLOCK_MAP applied to a minecraft blockid, meta pair.
    """
    CUBE_SIZE = 3
    SPHERE_RADIUS = 2
    fence = getattr(args, "fence", False)
    blocks = build_base_world(args.SL, args.H, args.GROUND_DEPTH, fence=fence)
    inst_segs = []
    shape_opts = {"SPHERE": {"radius": SPHERE_RADIUS}, "CUBE": {"size": CUBE_SIZE}}
    shapes = np.random.permutation(["SPHERE", "CUBE"])
    occupied_by_shapes = {}
    old_offset = [-100, -100, -100]
    for shape in shapes:
        opts = shape_opts[shape]
        opts["bid"] = bid()
        S = SHAPE_FNS[shape](**opts)
        m = np.round(np.mean([l for l, idm in S], axis=0)).astype("int32")
        offsets = np.random.randint(
            (0, args.GROUND_DEPTH, 0),
            (args.SL - CUBE_SIZE, args.H - CUBE_SIZE, args.SL - CUBE_SIZE),
        )
        count = 0
        while (
            abs(old_offset[0] - offsets[0]) + abs(old_offset[2] - offsets[2])
            < CUBE_SIZE + SPHERE_RADIUS
        ):
            offsets = np.random.randint(
                (0, args.GROUND_DEPTH, 0),
                (args.SL - CUBE_SIZE, args.H - CUBE_SIZE, args.SL - CUBE_SIZE),
            )
            count += 1
            assert (count < 100, "Is world too small? can't place shapes")
        old_offset = offsets
        inst_seg = []
        in_box = in_box_builder(0, 0, 0, args.SL, args.H, args.SL)
        record_shape(S, in_box, offsets, blocks, inst_seg, occupied_by_shapes)
        inst_segs.append({"tags": [shape], "locs": inst_seg})

    # not shifting y for gridworld
    o = (args.cuberite_x_offset, 0, args.cuberite_z_offset)
    blocks = [(l, idm) for l, idm in blocks.items()]
    blocks = shift(blocks, o)
    for i in inst_segs:
        i["locs"] = shift(i["locs"], o)

    return collect_scene(blocks, inst_segs, args)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--SL", type=int, default=SL)
    parser.add_argument("--H", type=int, default=H)
    parser.add_argument("--GROUND_DEPTH", type=int, default=GROUND_DEPTH)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--NUM_SCENES", type=int, default=3)
    parser.add_argument("--MAX_NUM_GROUND_HOLES", type=int, default=0)
    parser.add_argument("--fence", action="store_true", default=False)
    parser.add_argument("--cuberite_x_offset", type=int, default=-SL // 2)
    parser.add_argument("--cuberite_y_offset", type=int, default=63 - GROUND_DEPTH)
    parser.add_argument("--cuberite_z_offset", type=int, default=-SL // 2)
    parser.add_argument("--save_data_path", default="")
    parser.add_argument("--extra_simple", action="store_true", default=False)
    args = parser.parse_args()

    scenes = []
    for i in range(args.NUM_SCENES):
        if args.extra_simple:
            scenes.append(build_extra_simple_shape_scene(args))
        else:
            scenes.append(build_shape_scene(args))
    if args.save_data_path:
        with open(args.save_data_path, "w") as f:
            json.dump(scenes, f)
