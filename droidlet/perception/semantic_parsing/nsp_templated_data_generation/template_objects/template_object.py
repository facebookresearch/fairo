"""
Copyright (c) Facebook, Inc. and its affiliates.

This file defines the TemplateObject class and other data structures used across
template objects.
"""
from ..generate_utils import *
from ..tree_components import *


SCHEMATIC_TYPES = [
    RectanguloidShape,
    HollowRectanguloidShape,
    CubeShape,
    HollowCubeShape,
    SphereShape,
    HollowSphereShape,
    PyramidShape,
    RectangleShape,
    SquareShape,
    TriangleShape,
    CircleShape,
    DiskShape,
    EllipsoidShape,
    DomeShape,
    ArchShape,
    TowerShape,
    CategoryObject,
]

TAG_ADJECTIVES = [
    "round",
    "bright",
    "crooked",
    "steep",
    "blurry",
    "deep",
    "flat",
    "large",
    "tall",
    "broad",
    "fuzzy",
    "long",
    "narrow",
    "sleek",
    "sharp",
    "curved",
    "wide",
    "nice",
    "pretty",
]

TAG_NAMES = (
    [
        "box",
        "rectanguloid",
        "cube",
        "empty box",
        "hollow box",
        "hollow rectanguloid",
        "cube",
        "empty cube",
        "hollow cube",
        "ball",
        "sphere",
        "dome",
        "empty sphere",
        "empty ball",
        "hollow ball",
        "spherical shell",
        "hollow sphere",
        "pyramid",
        "rectangle",
        "square",
        "triangle",
        "circle",
        "disk",
        "ellipsoid",
        "dome",
        "arch",
        "tower",
        "wall",
    ]
    + MOBS
    + SUBCOMPONENT_LABELS
)


class TemplateObject:
    def __init__(self, node, template_attr):
        self.node = node
        self.template_attr = template_attr

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return


def get_template_names(obj, templ_index=0):
    return [type(temp_obj).__name__ for temp_obj in obj.node.template[templ_index]]
