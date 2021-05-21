"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import droidlet.base_util

"""This file has mappings from mob names to their ids and 
names of shapes to their functions"""
from typing import Dict, Callable
from droidlet.perception.craftassist import shapes

# mapping from canonicalized shape names to the corresponding functions
SPECIAL_SHAPE_FNS: Dict[str, Callable] = {
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
    "TOWER": shapes.tower,
}

# mapping from shape names to canonicalized shape names
SPECIAL_SHAPES_CANONICALIZE = {
    "rectanguloid": "RECTANGULOID",
    "box": "HOLLOW_RECTANGULOID",
    "empty box": "HOLLOW_RECTANGULOID",
    "hollow box": "HOLLOW_RECTANGULOID",
    "hollow rectanguloid": "HOLLOW_RECTANGULOID",
    "cube": "CUBE",
    "empty cube": "HOLLOW_CUBE",
    "hollow cube": "HOLLOW_CUBE",
    "ball": "SPHERE",
    "sphere": "SPHERE",
    "spherical shell": "SPHERICAL_SHELL",
    "empty sphere": "SPHERICAL_SHELL",
    "empty ball": "SPHERICAL_SHELL",
    "hollow sphere": "SPHERICAL_SHELL",
    "hollow ball": "SPHERICAL_SHELL",
    "pyramid": "PYRAMID",
    "rectangle": "RECTANGLE",
    "wall": "RECTANGLE",
    "slab": "RECTANGLE",
    "platform": "RECTANGLE",
    "square": "SQUARE",
    "flat wedge": "TRIANGLE",
    "triangle": "TRIANGLE",
    "circle": "CIRCLE",
    "disk": "DISK",
    "ellipsoid": "ELLIPSOID",
    "dome": "DOME",
    "arch": "ARCH",
    "archway": "ARCH",
    "stack": "TOWER",
    "tower": "TOWER",
}
