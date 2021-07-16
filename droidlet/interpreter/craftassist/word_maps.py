"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
"""This file has mappings from mob names to their ids and 
names of shapes to their functions"""

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
