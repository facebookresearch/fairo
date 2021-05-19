"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import droidlet.base_util

"""This file has helper functions for shapes.py"""
import random
import numpy as np

from droidlet.perception.craftassist import shapes

FORCE_SMALL = 10  # for debug

# Map shape name to function in shapes.py
SHAPE_FNS = {
    "CUBE": droidlet.base_util.cube,
    "HOLLOW_CUBE": shapes.hollow_cube,
    "RECTANGULOID": shapes.rectanguloid,
    "HOLLOW_RECTANGULOID": shapes.hollow_rectanguloid,
    "SPHERE": shapes.sphere,
    "SPHERICAL_SHELL": shapes.spherical_shell,
    "PYRAMID": shapes.square_pyramid,
    "SQUARE": shapes.square,
    "RECTANGLE": shapes.rectangle,
    "CIRCLE": shapes.circle,
    "DISK": shapes.disk,
    "TRIANGLE": shapes.triangle,
    "DOME": shapes.dome,
    "ARCH": shapes.arch,
    "ELLIPSOID": shapes.ellipsoid,
    "HOLLOW_TRIANGLE": shapes.hollow_triangle,
    "HOLLOW_RECTANGLE": shapes.hollow_rectangle,
    "RECTANGULOID_FRAME": shapes.rectanguloid_frame,
}

# list of shape names
SHAPE_NAMES = [
    "CUBE",
    "HOLLOW_CUBE",
    "RECTANGULOID",
    "HOLLOW_RECTANGULOID",
    "SPHERE",
    "SPHERICAL_SHELL",
    "PYRAMID",
    "SQUARE",
    "RECTANGLE",
    "CIRCLE",
    "DISK",
    "TRIANGLE",
    "DOME",
    "ARCH",
    "ELLIPSOID",
    "HOLLOW_TRIANGLE",
    "HOLLOW_RECTANGLE",
    "RECTANGULOID_FRAME",
]


def bid():
    allowed_blocks = {}
    allowed_blocks[1] = list(range(7))
    allowed_blocks[4] = [0]
    allowed_blocks[5] = list(range(6))
    allowed_blocks[12] = list(range(2))
    allowed_blocks[17] = list(range(4))
    allowed_blocks[18] = list(range(4))
    allowed_blocks[20] = [0]
    allowed_blocks[22] = [0]
    allowed_blocks[24] = list(range(3))
    allowed_blocks[34] = [0]
    allowed_blocks[35] = list(range(16))
    allowed_blocks[41] = [0]
    allowed_blocks[42] = [0]
    allowed_blocks[43] = list(range(8))
    allowed_blocks[45] = [0]
    allowed_blocks[48] = [0]
    allowed_blocks[49] = [0]
    allowed_blocks[57] = [0]
    allowed_blocks[95] = list(range(16))
    allowed_blocks[133] = [0]
    allowed_blocks[155] = list(range(3))
    allowed_blocks[159] = list(range(16))
    allowed_blocks[169] = [0]

    b = random.choice(list(allowed_blocks))
    m = random.choice(allowed_blocks[b])
    return (b, m)


def bernoulli(p=0.5):
    return np.random.rand() > p


def slope(ranges=10):
    return np.random.randint(1, ranges)


def sizes1(ranges=(1, 15)):
    ranges = list(ranges)
    ranges[1] = min(ranges[1], FORCE_SMALL)
    return np.random.randint(ranges[0], ranges[1])


def sizes2(ranges=(15, 15)):
    ranges = list(ranges)
    for i in range(2):
        ranges[i] = min(ranges[i], FORCE_SMALL)
    return (np.random.randint(1, ranges[0]), np.random.randint(1, ranges[1]))


def sizes3(ranges=(15, 15, 15)):
    ranges = list(ranges)
    for i in range(3):
        ranges[i] = min(ranges[i], FORCE_SMALL)
    return (
        np.random.randint(1, ranges[0]),
        np.random.randint(1, ranges[1]),
        np.random.randint(1, ranges[2]),
    )


def orientation2():
    return random.choice(["xy", "yz"])


def orientation3():
    return random.choice(["xy", "yz", "xz"])


def options_rectangle():
    return {"size": sizes2(), "orient": orientation3()}


def options_square():
    return {"size": sizes1(), "orient": orientation3()}


def options_triangle():
    return {"size": sizes1(), "orient": orientation3()}


def options_circle():
    return {"radius": sizes1(), "orient": orientation3()}


def options_disk():
    return {"radius": sizes1(), "orient": orientation3()}


def options_cube():
    return {"size": sizes1(ranges=(1, 8))}


def options_hollow_cube():
    return {"size": sizes1()}


def options_rectanguloid():
    return {"size": sizes3(ranges=(8, 8, 8))}


def options_hollow_rectanguloid():
    return {"size": sizes3()}


def options_sphere():
    return {"radius": sizes1(ranges=(3, 5))}


def options_spherical_shell():
    return {"radius": sizes1()}


def options_square_pyramid():
    return {"slope": slope(), "radius": sizes1()}


def options_tower():
    return {"height": sizes1(ranges=(1, 20)), "base": sizes1(ranges=(-3, 4))}


def options_ellipsoid():
    return {"size": sizes3()}


def options_dome():
    return {"radius": sizes1()}


def options_arch():
    return {"size": sizes1(), "distance": 2 * sizes1(ranges=(2, 5)) + 1}


def options_hollow_triangle():
    return {"size": sizes1() + 1, "orient": orientation3()}


def options_hollow_rectangle():
    return {"size": sizes1() + 1, "orient": orientation3()}


def options_rectanguloid_frame():
    return {"size": sizes3(), "only_corners": bernoulli()}


def shape_to_dicts(S):
    blocks = [
        {"x": s[0][0], "y": s[0][1], "z": s[0][2], "id": s[1][0], "meta": s[1][1]} for s in S
    ]
    return blocks


# Map shape name to option function
SHAPE_HELPERS = {
    "CUBE": options_cube,
    "HOLLOW_CUBE": options_hollow_cube,
    "RECTANGULOID": options_rectanguloid,
    "HOLLOW_RECTANGULOID": options_hollow_rectanguloid,
    "SPHERE": options_sphere,
    "SPHERICAL_SHELL": options_spherical_shell,
    "PYRAMID": options_square_pyramid,
    "SQUARE": options_square,
    "RECTANGLE": options_rectangle,
    "CIRCLE": options_circle,
    "DISK": options_disk,
    "TRIANGLE": options_triangle,
    "DOME": options_dome,
    "ARCH": options_arch,
    "ELLIPSOID": options_ellipsoid,
    "HOLLOW_TRIANGLE": options_hollow_triangle,
    "HOLLOW_RECTANGLE": options_hollow_rectangle,
    "RECTANGULOID_FRAME": options_rectanguloid_frame,
}
