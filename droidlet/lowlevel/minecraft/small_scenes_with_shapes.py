import numpy as np
import random
from droidlet.lowlevel.minecraft.shape_util import (
    SHAPE_NAMES,
    SHAPE_FNS,
    SHAPE_OPTION_FUNCTION_MAP,
)


# REMOVE ME WHEN THIS IS SHARED...:
BLOCK_MAP = {
    (0, 0): 0,  # Air
    (1, 0): 1,  # Stone
    (1, 1): 2,  # Granite
    (1, 2): 3,  # Polished Granite
    (1, 3): 4,  # Diorite
    (1, 4): 5,  # Polished Diorite
    (1, 5): 6,  # Andesite
    (1, 6): 7,  # Polished Andesite
    (2, 0): 8,  # Grass
    (3, 0): 9,  # Dirt
    (3, 1): 10,  # Coarse Dirt
    (3, 2): 11,  # Podzol
    (4, 0): 12,  # Cobblestone
    (5, 0): 13,  # Oak Wood Plank
    (5, 1): 14,  # Spruce Wood Plank
    (5, 2): 15,  # Birch Wood Plank
    (5, 3): 16,  # Jungle Wood Plank
    (5, 4): 17,  # Acacia Wood Plank
    (5, 5): 18,  # Dark Oak Wood Plank
    (6, 0): 19,  # Oak Sapling
    (6, 1): 20,  # Spruce Sapling
    (6, 2): 21,  # Birch Sapling
    (6, 3): 22,  # Jungle Sapling
    (6, 4): 23,  # Acacia Sapling
    (6, 5): 24,  # Dark Oak Sapling
    (7, 0): 25,  # Bedrock
    (8, 0): 26,  # Flowing Water
    (9, 0): 27,  # Still Water
    (12, 0): 28,  # Sand
    (12, 1): 29,  # Red Sand
    (13, 0): 30,  # Gravel
    (17, 0): 31,  # Oak Wood
    (17, 1): 32,  # Spruce Wood
    (17, 2): 33,  # Birch Wood
    (17, 3): 34,  # Jungle Wood
    (18, 0): 35,  # Oak Leaves
    (18, 1): 36,  # Spruce Leaves
    (18, 2): 37,  # Birch Leaves
    (18, 3): 38,  # Jungle Leaves
    (24, 0): 39,  # Sandstone
    (24, 1): 40,  # Chiseled Sandstone
    (24, 2): 41,  # Smooth Sandstone
    (31, 0): 42,  # Dead Shrub
    (31, 1): 43,  # Grass
    (31, 2): 44,  # Fern
    (32, 0): 45,  # Dead Bush
    (35, 0): 46,  # White Wool
    (35, 1): 47,  # Orange Wool
    (35, 2): 48,  # Magenta Wool
    (35, 3): 49,  # Light Blue Wool
    (35, 4): 50,  # Yellow Wool
    (35, 5): 51,  # Lime Wool
    (35, 6): 52,  # Pink Wool
    (35, 7): 53,  # Gray Wool
    (35, 8): 54,  # Light Gray Wool
    (35, 9): 55,  # Cyan Wool
    (35, 10): 56,  # Purple Wool
    (35, 11): 57,  # Blue Wool
    (35, 12): 58,  # Brown Wool
    (35, 13): 59,  # Green Wool
    (35, 14): 60,  # Red Wool
    (35, 15): 61,  # Black Wool
    (37, 0): 62,  # Dandelion
    (38, 0): 63,  # Poppy
    (39, 0): 64,  # Brown Mushroom
    (40, 0): 65,  # Red Mushroom
}


def bid():
    return (35, np.random.randint(16))


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


def build_base_world(sl, h, g):
    W = []
    for i in range(sl):
        for j in range(g):
            for k in range(sl):
                W.append(((i, j, k), white()))
    return W


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
    blocks = build_base_world(args.SL, args.H, args.GROUND_DEPTH)
    num_shapes = np.random.randint(1, args.MAX_NUM_SHAPES + 1)
    for t in range(num_shapes):
        shape = random.choice(SHAPE_NAMES)
        opts = SHAPE_OPTION_FUNCTION_MAP[shape]()
        opts["bid"] = bid()
        S = SHAPE_FNS[shape](**opts)
        offsets = np.random.randint((args.SL, args.H, args.SL))
        offsets[1] = offsets[1] - args.GROUND_DEPTH
        for l, idm in S:
            ln = np.add(l, offsets)
            if ln[0] < args.SL and ln[1] >= 0 and ln[1] < args.H and ln[2] < args.SL:
                blocks.append((l, idm))
    J = {}
    J["avatarInfo"] = {"pos": avatar_pos(args, blocks), "look": avatar_look(args, blocks)}
    J["agentInfo"] = {"pos": agent_pos(args, blocks), "look": agent_look(args, blocks)}
    J["schematic_for_cuberite"] = [
        {
            "x": l[0] + args.cuberite_x_offset,
            "y": l[1] + args.cuberite_y_offset,
            "z": l[2] + args.cuberite_z_offset,
            "id": idm[0],
            "meta": idm[1],
        }
        for l, idm in blocks
    ]
    transformed_blocks = [(l[0], l[1], l[2], BLOCK_MAP[idm]) for l, idm in blocks]
    J["blocks"] = transformed_blocks
    return J


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--SL", type=int, default=13)
    parser.add_argument("--H", type=int, default=9)
    parser.add_argument("--GROUND_DEPTH", type=int, default=5)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--NUM_SCENES", type=int, default=3)
    parser.add_argument("--cuberite_x_offset", type=int, default=-13 // 2)
    parser.add_argument("--cuberite_y_offset", type=int, default=63 - 5)
    parser.add_argument("--cuberite_z_offset", type=int, default=-13 // 2)
    parser.add_argument("--save_data_path", default="")
    args = parser.parse_args()

    scenes = []
    for i in range(args.NUM_SCENES):
        scenes.append(build_shape_scene(args))
    if args.NUM_SCENES == 1:
        scenes = scenes[0]
    if args.save_data_path:
        with open(args.save_data_path, "w") as f:
            json.dump(scenes, f)
