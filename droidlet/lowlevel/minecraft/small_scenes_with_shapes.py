import numpy as np
import random
from droidlet.lowlevel.minecraft.shape_util import SHAPE_NAMES, SHAPE_FNS, SHAPE_OPTION_FUNCTION_MAP

def bid():
    return (35, np.random.randint(16))

def build_base_world(sl, h, g):
    W = np.zeros((sl, h, sl, 2))
    for i in range(g):
        for j in range(sl):
            for k in range(sl):
                W[j, i, k, 0] = 42
                W[j, i, k, 1] = 0
    return W

def build_shape_scene(args):
    """
    Build a scene using basic shapes,
    outputs an SLxHxSLx2 numpy array corresponding to x, y, z, (bid, meta)
    """
    W = build_base_world(args.SL, args.H, args.GROUND_DEPTH) 
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
                W[ln[0], ln[1], ln[2], 0] = idm[0]
                W[ln[0], ln[1], ln[2], 1] = idm[1]

    return W


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--SL", type=int, default=12)
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--GROUND_DEPTH", type=int, default=3)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--NUM_SCENES", type=int, default=1)
    parser.add_argument("--save_data_path", default="")
    args = parser.parse_args()

    scenes = []
    for i in range(args.NUM_SCENES):
        scenes.append(build_shape_scene(args).tolist())
    if args.save_data_path:
        with open(args.save_data_path, "w") as f:
            json.dump(scenes, f)