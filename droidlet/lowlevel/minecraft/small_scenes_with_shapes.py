"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import pickle
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


# FIXME! better control of distribution and put this in a different file
# also no control of cuberite coloring
def random_mob_color(mobname):
    if mobname == "rabbit":
        return np.random.choice(["brown", "white", "black", "mottled", "gray"])
    if mobname == "pig":
        return np.random.choice(["brown", "white", "black", "mottled", "pink"])
    if mobname == "chicken":
        return np.random.choice(["white", "yellow", "brown"])
    if mobname == "sheep":
        return np.random.choice(["brown", "white", "black", "mottled", "gray"])
    if mobname == "cow":
        return np.random.choice(["brown", "white", "black", "mottled", "gray"])
    return "white"


def parse_and_execute_mob_config(args):
    """
    mob_config can be a dict of the form
    {"mobs": [{"mobtype": MOBTYPE,  "pose": (x, y, z, pitch, yaw), "color": COLOR}, ...]}
    or
    {"mob_generator": CALLABLE}
    or
    {"num_mobs": INT, "mob_probs":{MOBNAME:FLOAT_LOGIT, ..., MOBNAME:FLOAT_LOGIT}}
    if "mob_probs" is not specified, it is uniform over ["rabbit", "cow", "pig", "chicken", "sheep"]
    or string of the form
    num_mobs:x;mobname:float;mobname:float;...
    the floats are the probability of sampling that mob

    returns a list of [{"mobtype": MOBTYPE,  "pose": (x, y, z, pitch, yaw), "color": COLOR}, ...]
    """

    mob_config = args.mob_config
    if type(mob_config) is str:
        c = {}
        o = mob_config.split(";")
        if len(o) > 0:
            try:
                assert o[0].startswith("num_mobs")
                num_mobs = int(o[0].split(":")[1])
                c["num_mobs"] = num_mobs
                c["mob_probs"] = {}
                for p in o[1:]:
                    name, prob = p.split(":")
                    c["mob_probs"][name] = float(prob)
            except:
                c = {}
    else:
        c = mob_config
    assert type(c) is dict
    if not c:
        return []
    elif c.get("mobs"):
        return c["mobs"]
    elif c.get("mob_generator"):
        return c["mob_generator"](args)
    elif c.get("num_mobs"):
        num_mobs = c.get("num_mobs")
        if not c["mob_probs"]:
            md = [("rabbit", "cow", "pig", "chicken", "sheep"), (1.0, 1.0, 1.0, 1.0, 1.0)]
        else:
            md = list(zip(*c["mob_probs"].items()))
        probs = np.array(md[1])
        assert probs.min() >= 0.0
        probs = probs / probs.sum()
        mobnames = np.random.choice(md[0], size=num_mobs, p=probs).tolist()
        mobs = []
        for m in mobnames:
            # mob pose set in collect_scene for now if not specified
            mobs.append({"mobtype": m, "pose": None, "color": random_mob_color(m)})
        return mobs
    else:
        raise Exception("malformed mob opts {}".format(c))


def bid(nowhite=True):
    if nowhite:
        return (35, np.random.randint(15) + 1)
    else:
        return (35, np.random.randint(16))


def red():
    return (35, 14)


def white():
    return (35, 0)


def dirt():
    return (3, 0)


def grass():
    return (2, 0)


def make_pose(args, loc=None, pitchyaw=None, height_map=None):
    """
    make a random pose for player or mob.
    if loc or pitchyaw is specified, use those
    if height_map is specified, finds a point close to the loc
        1 block higher than the height_map, but less than ENTITY_HEIGHT from
        args.H
    TODO option to input object locations and pick pitchyaw to look at one
    """
    ENTITY_HEIGHT = 2
    if loc is None:
        x, y, z = np.random.randint((args.SL / 3, args.H / 3, args.SL / 3)) + args.SL / 3
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
        okh = np.array(np.nonzero(height_map < args.H - ENTITY_HEIGHT))
        if okh.shape[1] == 0:
            raise Exception(
                "no space for entities, height map goes up to args.H-ENTITY_HEIGHT everywhere"
            )
        d = np.linalg.norm((okh - np.array((x, z)).reshape(2, 1)), 2, 0)
        minidx = np.argmin(d)
        x = int(okh[0, minidx])
        z = int(okh[1, minidx])
        y = int(height_map[x, z] + 1)
    return x, y, z, pitch, yaw


def build_base_world(sl, h, g, fence=False):
    W = {}
    for i in range(sl):
        for j in range(g):
            for k in range(sl):
                if (
                    (i < sl / 3 or i >= 2 * sl / 3 or k < sl / 3 or k >= 2 * sl / 3)
                    and j == g - 1
                    and fence
                ):
                    idm = red()
                elif j == (g - 1):
                    idm = grass()
                else:
                    idm = dirt()
                W[(i, j, k)] = idm
    return W


def shift_list(blocks_list, s):
    for i in range(len(blocks_list)):
        b = blocks_list[i]
        if len(b) == 2:
            l, idm = b
            blocks_list[i] = ((l[0] + s[0], l[1] + s[1], l[2] + s[2]), idm)
        else:
            assert len(b) == 3
            blocks_list[i] = (b[0] + s[0], b[1] + s[1], b[2] + s[2])
    return blocks_list


def shift_dict(block_dict, s):
    out = {}
    for l, idm in block_dict.items():
        out[(l[0] + s[0], l[1] + s[1], l[2] + s[2])] = idm
    return out


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


def collect_scene(blocks, inst_segs, args, mobs=[]):
    J = {}
    J["inst_seg_tags"] = inst_segs
    mapped_blocks = [
        (int(l[0]), int(l[1]), int(l[2]), int(IGLU_BLOCK_MAP[idm])) for l, idm in blocks.items()
    ]
    J["blocks"] = mapped_blocks

    # FIXME not shifting positions of agents and mobs properly for cuberite
    # FIXME not using the mob positions in cuberite...
    height_map = np.zeros((args.SL, args.SL))
    for l, idm in blocks.items():
        if l[1] > height_map[l[0], l[2]] and idm[0] > 0:
            height_map[l[0], l[2]] = l[1]
    J["mobs"] = []
    for mob in mobs:
        if mob.get("pose"):
            x, y, z, p, yaw = mob["pose"]
            loc = (x, y, z)
            pitchyaw = (p, yaw)
        else:
            loc = None
            # For mobs we want random yaw, but 0 pitch
            yaw = np.random.uniform(-np.pi, np.pi)
            pitchyaw = (0, yaw)
        mob["pose"] = make_pose(args, loc=loc, pitchyaw=pitchyaw, height_map=height_map)
        J["mobs"].append(mob)
    # FIXME not using the avatar and agent position in cuberite...

    x, y, z, p, yaw = make_pose(args, height_map=height_map)
    J["avatarInfo"] = {"pos": (x, y, z), "look": (p, yaw)}
    x, y, z, p, yaw = make_pose(args, height_map=height_map)
    J["agentInfo"] = {"pos": (x, y, z), "look": (p, yaw)}

    o = (0, args.cuberite_y_offset, 0)
    blocks = shift_dict(blocks, o)
    J["schematic_for_cuberite"] = [
        {"x": int(l[0]), "y": int(l[1]), "z": int(l[2]), "id": int(idm[0]), "meta": int(idm[1])}
        for l, idm in blocks.items()
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
    if args.iglu_scenes:
        with open(args.iglu_scenes, "rb") as f:
            assets = pickle.load(f)
            sid = np.random.choice(list(assets.keys()))
            scene = assets[sid]
            scene = scene.transpose(1, 0, 2)
            for i in range(11):
                for j in range(9):
                    for k in range(11):
                        h = j + args.GROUND_DEPTH
                        if h < args.H:
                            # TODO? fix colors
                            # TODO: assuming  this world bigger in xz than iglu
                            c = scene[i, j, k] % 16
                            if c > 0:
                                blocks[(i + 1, h, k + 1)] = (35, c)

    num_shapes = np.random.randint(0, args.MAX_NUM_SHAPES + 1)
    occupied_by_shapes = {}
    inst_segs = []
    for t in range(num_shapes):
        shape = np.random.choice(SHAPE_NAMES)
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
        shape = np.random.choice(HOLE_NAMES)
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
    mobs = parse_and_execute_mob_config(args)
    J = {}
    # not shifting y for gridworld
    o = (args.cuberite_x_offset, 0, args.cuberite_z_offset)
    blocks = shift_dict(blocks, o)
    for i in inst_segs:
        i["locs"] = shift_list(i["locs"], o)

    return collect_scene(blocks, inst_segs, args, mobs=mobs)


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
            assert count < 100, "Is world too small? can't place shapes"
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
    parser.add_argument("--mob_config", type=str, default="")
    parser.add_argument("--GROUND_DEPTH", type=int, default=GROUND_DEPTH)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--NUM_SCENES", type=int, default=3)
    parser.add_argument("--MAX_NUM_GROUND_HOLES", type=int, default=0)
    parser.add_argument("--fence", action="store_true", default=False)
    parser.add_argument("--cuberite_x_offset", type=int, default=-SL // 2)
    parser.add_argument("--cuberite_y_offset", type=int, default=63 - GROUND_DEPTH)
    parser.add_argument("--cuberite_z_offset", type=int, default=-SL // 2)
    parser.add_argument("--save_data_path", default="")
    parser.add_argument("--iglu_scenes", default="")
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
