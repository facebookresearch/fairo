"""
Copyright (c) Facebook, Inc. and its affiliates.

This file has the definitions and properties of the components that go into the
action tree.
Following is a list of component nodes:
- Schematic, that can be of type:
    - CategoryObject
    - Shape, that can be of type:
        - BlockShape
        - RectanguloidShape
        - HollowRectanguloidShape
        - CubeShape
        - HollowCubeShape
        - SphereShape
        - HollowSphereShape
        - PyramidShape
        - RectangleShape
        - SquareShape
        - TriangleShape
        - CircleShape
        - DiskShape
        - EllipsoidShape
        - DomeShape
        - ArchShape
        - TowerShape
- Location, that can be of type:
    - Coordinates
    - LocationDelta
    - SpeakerLook
    - SpeakerPos
    - AgentPos
- BlockObject, that can be of type:
    - Object
    - PointedObject
- Mob
"""
import os
import copy
import random

from collections import OrderedDict

from .generate_utils import *


SUBCOMPONENT_LABELS = [
    "wall",
    "roof",
    "window",
    "foundation",
    "door",
    "floor",
    "ceiling",
    "siding",
    "column",
    "porch",
    "entrance",
    "corner",
    "balcony",
    "railing",
    "decoration",
    "support",
    "base",
    "pillar",
    "beam",
    "step",
    "ledge",
    "fence",
    "facade",
    "overhang",
    "footing",
    "walkway",
    "stairs",
    "basement",
    "chimney",
    "doorway",
    "cornerstone",
    "deck",
    "side",
    "eave",
    "sill",
    "patio",
    "frame",
    "steps",
    "windowsill",
    "post",
    "header",
    "pane",
    "ridge",
    "roofing",
    "awning",
    "rooftop",
    "trim",
    "stair",
    "gable",
    "garden",
    "light",
    "brick",
    "edge",
    "fascia",
    "entryway",
    "gutter",
    "platform",
    "panel",
    "foot",
    "ground",
    "trap",
    "frontage",
    "storage",
    "torch",
    "crawlspace",
    "soffit",
    "eaves",
    "tower",
    "parapet",
    "jamb",
    "attic",
    "staircase",
    "skylight",
    "barrier",
    "glass",
    "terrace",
    "room",
    "doorstep",
    "pier",
    "ladder",
    "bedroom",
    "cladding",
    "partition",
    "kitchen",
    "doorknob",
    "stairway",
    "opening",
    "shutter",
    "exterior",
    "strut",
    "garage",
    "glazing",
    "shingles",
    "stoop",
    "yard",
    "sidewalk",
    "rail",
    "casing",
    "substructure",
    "paneling",
]

MOBS = [
    "elder guardian",
    "wither skeleton",
    "stray",
    "husk",
    "zombie villager",
    "skeleton horse",
    "zombie horse",
    "donkey",
    "mule",
    "evoker",
    "vex",
    "vindicator",
    "creeper",
    "skeleton",
    "spider",
    "zombie",
    "slime",
    "ghast",
    "zombie pigman",
    "enderman",
    "cave spider",
    "silverfish",
    "blaze",
    "magma cube",
    "bat",
    "witch",
    "endermite",
    "guardian",
    "pig",
    "sheep",
    "cow",
    "chicken",
    "squid",
    "mooshroom",
    "horse",
    "rabbit",
    "polar bear",
    "llama",
    "parrot",
    "villager",
    "ocelot",
    "wolf",
    "shulker",
]

BLOCK_TYPES = [
    "air",
    "stone",
    "granite",
    "polished granite",
    "diorite",
    "polished diorite",
    "andesite",
    "polished andesite",
    "dirt",
    "coarse dirt",
    "podzol",
    "cobblestone",
    "oak wood plank",
    "spruce wood plank",
    "birch wood plank",
    "jungle wood plank",
    "acacia wood plank",
    "dark oak wood plank",
    "oak sapling",
    "spruce sapling",
    "birch sapling",
    "jungle sapling",
    "acacia sapling",
    "dark oak sapling",
    "bedrock",
    "flowing water",
    "still water",
    "flowing lava",
    "still lava",
    "sand",
    "red sand",
    "gravel",
    "gold ore",
    "iron ore",
    "coal ore",
    "oak wood",
    "spruce wood",
    "birch wood",
    "jungle wood",
    "oak leaves",
    "spruce leaves",
    "birch leaves",
    "jungle leaves",
    "sponge",
    "wet sponge",
    "glass",
    "lapis lazuli ore",
    "lapis lazuli block",
    "dispenser",
    "sandstone",
    "chiseled sandstone",
    "smooth sandstone",
]

ABSTRACT_SIZE = [
    "really tiny",
    "very tiny",
    "tiny",
    "really small",
    "very small",
    "really little",
    "very little",
    "small",
    "little",
    "medium",
    "medium sized",
    "big",
    "large",
    "really big",
    "very big",
    "really large",
    "very large",
    "huge",
    "gigantic",
    "really huge",
    "very huge",
    "really gigantic",
    "very gigantic",
]

COLOURS = ["red", "blue", "green", "yellow", "purple", "grey", "brown", "black", "orange"]

with open(os.path.join(os.path.dirname(__file__), "categories.txt")) as f:
    CATEGORIES = [line.strip() for line in f.readlines()] + SUBCOMPONENT_LABELS

SHAPE_NAMES = [
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
    "platform",
    "slab",
]

CONCRETE_OBJECT_NAMES = SHAPE_NAMES + SUBCOMPONENT_LABELS + CATEGORIES

CONDITION_TYPES = ["ADJACENT_TO_BLOCK_TYPE", "NEVER"]


class ComponentNode:
    """This class is a node in the action tree and represents the components of the
    tree that are not Actions.

    A node can have a list of node types, it can be (CHOICES).

    generate() : is responsible for initializing the CHOICES.
    generate_description() : Generates the natural language description.
    to_dict() : Generates the action tree recursively over the children.
    """

    CHOICES = None  # a list of node types that can be substituted for this node

    def __init__(self, template_attr={}):
        self.args = None  # populated by self.generate()
        self.description = None  # populated by self.generate_description()
        self._action_description = None
        self._template_attr = template_attr

    def generate_description(self):
        if self.description is None:
            self.description = self._generate_description()
        return self.description

    @classmethod
    def generate(cls):
        if cls.CHOICES:
            c = random.choice(cls.CHOICES)
            return c.generate()

        return cls()

    def __repr__(self):
        if self.args:
            return "<{} ({})>".format(type(self).__name__, ", ".join(map(str, self.args)))
        else:
            return "<{}>".format(type(self).__name__)

    def to_dict(self):
        d = {}

        if hasattr(self, "location_type") and type(self.location_type) == type:
            self.location_type = to_snake_case(self.location_type.__name__, case="upper")
            if self.location_type in ["BLOCK_OBJECT", "MOB"]:
                self.location_type = "REFERENCE_OBJECT"

        # For each recursive child, pass along the description of topmost action.
        if getattr(self, "_location", None) is not None:
            self._location._action_description = self._action_description
            d["location"] = self._location.to_dict()

        if getattr(self, "_reference_object", None) is not None:
            self._reference_object._action_description = self._action_description
            d["reference_object"] = self._reference_object.to_dict()

        if getattr(self, "_block_object", None) is not None:
            self._block_object._action_description = self._action_description
            d["reference_object"] = self._block_object.to_dict()

        if getattr(self, "_block_object_1", None) is not None:
            self._block_object_1._action_description = self._action_description
            d["reference_object_1"] = self._block_object_1.to_dict()

        if getattr(self, "_block_object_2", None) is not None:
            self._block_object_2._action_description = self._action_description
            d["reference_object_2"] = self._block_object_2.to_dict()

        if getattr(self, "_repeat", None):
            self._repeat._action_description = self._action_description
            # For between or any repeats that only need plural of names
            # don't add repeat dict
            if self._repeat.to_dict()["repeat_key"] != "ALL_ONLY":
                d["repeat"] = self._repeat.to_dict()

        if getattr(self, "_mob", None) is not None:
            self._mob._action_description = self._action_description
            d["reference_object"] = self._mob.to_dict()

        if getattr(self, "_mob_1", None) is not None:
            self._mob_1._action_description = self._action_description
            d["reference_object_1"] = self._mob_1.to_dict()

        if getattr(self, "_mob_2", None) is not None:
            self._mob_2._action_description = self._action_description
            d["reference_object_2"] = self._mob_2.to_dict()

        # fix reference object filters
        for key in ["reference_object", "reference_object_1", "reference_object_2"]:
            if key in d:
                val = d[key]
                # if "repeat" in val:
                #     d[key] = {"repeat": val["repeat"]}
                #     val.pop("repeat")
                #     d[key]["filters"] = val
                # else:
                d[key] = {"filters": val}

        if getattr(self, "_memory_data", None) is not None:
            self._memory_data._action_description = self._action_description
            d["memory_data"] = self._memory_data.to_dict()

        for attr, val in self.__dict__.items():
            if (
                not attr.startswith("_")
                and val not in (None, "")
                and attr != "args"
                and attr != "description"
                and attr != "template"
            ):
                d[attr] = val
                if (attr.startswith("has_")) or (
                    attr
                    in [
                        "coordinates",
                        "steps",
                        "block_type",
                        "repeat_count",
                        "target_action_type",
                    ]
                ):
                    if type(val) == str and val.startswith("_"):
                        continue
                    span = find_span(self._action_description, val)
                    d[attr] = span
        # Put all 'has_' under triples
        old_d = copy.deepcopy(d)
        for key, val in old_d.items():
            if key.startswith("has_"):
                if "triples" not in d:
                    d["triples"] = []
                triples_dict = {"pred_text": key, "obj_text": val}
                d["triples"].append(triples_dict)
                d.pop(key)

        # fix location type now
        if "location_type" in d:
            value = d["location_type"]
            if value in ["SPEAKER_LOOK", "AGENT_POS", "SPEAKER_POS", "COORDINATES"]:
                updated_value = value  # same for coordinates and speaker_look
                if value == "AGENT_POS":
                    updated_value = "AGENT"
                elif value == "SPEAKER_POS":
                    updated_value = "SPEAKER"
                elif value == "COORDINATES":
                    updated_value = {"coordinates_span": d["coordinates"]}

                # add to reference object instead
                if "reference_object" in d:
                    d["reference_object"]["special_reference"] = updated_value
                else:
                    d["reference_object"] = {"special_reference": updated_value}
            d.pop("location_type")
            d.pop("coordinates", None)

        return d


###############
## SCHEMATIC ##
###############


class CategoryObject(ComponentNode):
    """CategoryObject is picked from a list of objects we have from the
    minecraft_specs folder.
    __init__ (): Pick the object, assign name and block_type.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(template_attr=template_attr)
        cat_object = random.choice(self._template_attr.get("non_shape_names", CATEGORIES))
        if repeat_key:
            cat_object = make_plural(cat_object)

        self.has_name = cat_object
        self.has_block_type = (
            random.choice(self._template_attr.get("block_types", BLOCK_TYPES))
            if block_type
            else None
        )

    def _generate_description(self):
        out_dict = {}
        out_dict["word"] = self.has_name
        if self.has_block_type:
            out_dict["block_type"] = self.has_block_type

        return out_dict


class Shape(ComponentNode):
    """Shape is a superclass for different Shape types like: Cube, rectanguloid,
    sphere etc.
    __init__(): Picks the block_type. The specific shape attributes are assigned in the individual
                child classes.
    """

    KEYS = [
        "has_size",
        "has_thickness",
        "has_radius",
        "has_height",
        "has_slope",
        "has_orientation",
        "has_distance",
        "has_base",
    ]

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(template_attr=template_attr)
        self.has_block_type = (
            random.choice(self._template_attr.get("block_types", BLOCK_TYPES))
            if block_type
            else None
        )
        self.has_name = None

    def _generate_description(self):
        of = random.choice(["of", "with"])
        attrs = []
        for key in self.KEYS:
            val = getattr(self, key, None)
            if val is None:
                continue
            if type(val) in (list, tuple):
                # For has_size format the text as: a b c
                # a x b x c or a by b by c
                if key == "has_size":
                    val = random.choice(
                        [
                            " ".join(map(str, val)),
                            " x ".join(map(str, val)),
                            " by ".join(map(str, val)),
                        ]
                    )

            key = key[4:]  # extract "size" from "has_size"
            attrs.append("{} {} {}".format(of, key, val))
        random.shuffle(attrs)

        out_dict = {}
        out_dict["word"] = self._word
        if attrs:
            out_dict["shape_attributes"] = attrs
        if self.has_block_type:
            out_dict["block_type"] = self.has_block_type

        return out_dict


class BlockShape(Shape):
    """Subclass of Shape represents a single block.
    __init__(): Assigning shape type, size and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type)
        # block is a 1 X 1 X 1 rectanguloid
        """schematic_attributes specifies whether any 1x3 size can be assigned or height, width etc
        are assigned separately one at a time.
        """
        self._word = random.choice(["block", "square"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class RectanguloidShape(Shape):
    """Subclass of Shape represents a rectanguloid.
    __init__(): Assigning shape type, size and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        # rectanguloid is width x height x thickness
        """schematic_attributes specifies whether any 1x3 size can be assigned or height, width etc
        are assigned separately one at a time.
        """
        if schematic_attributes:
            if type(schematic_attributes) == bool:
                self.has_size = random.sample(
                    self._template_attr.get("size", range(3, 51)), 3
                )  # pick 3 random numbers
            elif type(schematic_attributes) == dict:  # assign only what's specified.
                if "height" in schematic_attributes:
                    self.has_height = schematic_attributes["height"]
                if "width" in schematic_attributes:
                    self.has_width = schematic_attributes["width"]
        self._word = random.choice(["rectanguloid"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class HollowRectanguloidShape(Shape):
    """Subclass of Shape, represents a hollow rectanguloid.
    __init__(): Assigning shape type, size, thickness and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if (
            schematic_attributes
        ):  # schematic_attributes specifies that the size and thickness have to be assigned
            self.has_size = random.sample(self._template_attr.get("size", range(3, 51)), 3)
            self.has_thickness = (
                random.choice(self._template_attr.get("thickness", range(1, 6)))
                if pick_random()
                else None
            )
        self._word = random.choice(["box", "empty box", "hollow box", "hollow rectanguloid"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class CubeShape(Shape):
    """Subclass of Shape, represents a cube.
    __init__(): Assigning shape type, size and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_size = (
                random.choice(self._template_attr.get("size", range(3, 51)))
                if schematic_attributes
                else None
            )
        self._word = random.choice(["cube"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class HollowCubeShape(Shape):
    """Subclass of Shape, represents a hollow cube.
    __init__(): Assigning shape type, size, thickness and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_size = random.choice(self._template_attr.get("size", range(3, 51)))
            self.has_thickness = (
                random.choice(self._template_attr.get("thickness", range(1, 6)))
                if pick_random()
                else None
            )
        self._word = random.choice(["empty cube", "hollow cube"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class SphereShape(Shape):
    """Subclass of Shape, represents a sphere.
    __init__(): Assigning shape type, radius and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_radius = random.choice(self._template_attr.get("radius", range(3, 51)))
        self._word = random.choice(["ball", "sphere"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class HollowSphereShape(Shape):
    """Subclass of Shape, repesents a hollow sphere.
    __init__(): Assigning shape type, radius, thickness and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_thickness = random.choice(self._template_attr.get("thickness", range(1, 6)))
            self.has_radius = (
                random.choice(self._template_attr.get("radius", range(3, 51)))
                if pick_random()
                else None
            )
        self._word = random.choice(
            ["empty sphere", "empty ball", "hollow ball", "spherical shell", "hollow sphere"]
        )
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class PyramidShape(Shape):
    """Subclass of Shape, repesents a pyramid.
    __init__(): Assigning shape type, radius, height, slope and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_radius = random.choice(self._template_attr.get("radius", range(3, 51)))
            self.has_height = (
                random.choice(self._template_attr.get("height", range(3, 51)))
                if pick_random()
                else None
            )
            self.has_slope = (
                random.choice(self._template_attr.get("slope", range(1, 11)))
                if pick_random()
                else None
            )
        self._word = random.choice(["pyramid"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class RectangleShape(Shape):
    """Subclass of Shape, repesents a rectangle.
    __init__(): Assigning shape type, size, orientation type and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            if type(schematic_attributes) == bool:
                self.has_size = random.sample(self._template_attr.get("size", range(3, 51)), 2)
                self.has_orientation = random.choice(["xy", "yz", "xz"]) if pick_random() else None
            elif type(schematic_attributes) == dict:
                if "height" in schematic_attributes:
                    self.has_height = schematic_attributes["height"]
                if "length" in schematic_attributes:
                    self.has_length = schematic_attributes["length"]
        self._word = random.choice(["rectangle", "wall", "slab", "platform"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class SquareShape(Shape):
    """Subclass of Shape, repesents a square.
    __init__(): Assigning shape type, size, orientation type and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_size = random.choice(self._template_attr.get("size", range(3, 51)))
            self.has_orientation = random.choice(["xy", "yz", "xz"]) if pick_random() else None
        self._word = random.choice(["square"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class TriangleShape(Shape):
    """Subclass of Shape, represents an equilateral triangle.
    __init__(): Assigning shape type, size, orientation type, thickness and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_size = random.choice(self._template_attr.get("size", range(3, 11)))
            self.has_orientation = random.choice(["xy", "yz", "xz"]) if pick_random() else None
            self.has_thickness = (
                random.choice(self._template_attr.get("thickness", range(1, 6)))
                if pick_random()
                else None
            )
        self._word = random.choice(["triangle"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class CircleShape(Shape):
    """Subclass of Shape, repesents a circle.
    __init__(): Assigning shape type, radius, orientation type, thickness and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_radius = random.choice(self._template_attr.get("radius", range(3, 51)))
            self.has_orientation = random.choice(["xy", "yz", "xz"]) if pick_random() else None
            self.has_thickness = (
                random.choice(self._template_attr.get("thickness", range(1, 6)))
                if pick_random()
                else None
            )
        self._word = random.choice(["circle"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class DiskShape(Shape):
    """Subclass of Shape, represents a disk.
    __init__(): Assigning shape type, radius, orientation type, thickness and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_radius = random.choice(self._template_attr.get("radius", range(3, 51)))
            self.has_orientation = random.choice(["xy", "yz", "xz"]) if pick_random() else None
            self.has_thickness = (
                random.choice(self._template_attr.get("thickness", range(1, 6)))
                if pick_random()
                else None
            )
        self._word = random.choice(["disk"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class EllipsoidShape(Shape):
    """Subclass of Shape, represents an ellipsoid.
    __init__(): Assigning shape type, size, and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        self.has_size = (
            random.sample(self._template_attr.get("size", range(3, 51)), 3)
            if schematic_attributes
            else None
        )
        self._word = random.choice(["ellipsoid"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class DomeShape(Shape):
    """Subclass of Shape, repesents a dome.
    __init__(): Assigning shape type, radius, thickness and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_radius = random.choice(self._template_attr.get("radius", range(10, 51)))
            self.has_orientation = random.choice(["xy", "yz", "xz"]) if pick_random() else None
            self.has_thickness = (
                random.choice(self._template_attr.get("thickness", range(1, 4)))
                if pick_random()
                else None
            )
        self._word = random.choice(["dome"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class ArchShape(Shape):
    """Subclass of Shape, repesents an arch.
    __init__(): Assigning shape type, size, orientation type, distance and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            self.has_size = random.choice(self._template_attr.get("size", range(3, 51)))
            self.has_orientation = random.choice(["xy", "xz"]) if pick_random() else None
            self.has_distance = (
                random.choice(self._template_attr.get("distance", range(1, 50, 2)))
                if pick_random()
                else None
            )
        self._word = random.choice(["arch", "archway"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class TowerShape(Shape):
    """Subclass of Shape, represents a tower.
    __init__(): Assigning shape type, size, orientation type, distance and word.
    """

    def __init__(
        self, block_type=False, schematic_attributes=False, repeat_key=None, template_attr={}
    ):
        super().__init__(block_type=block_type, template_attr=template_attr)
        if schematic_attributes:
            if type(schematic_attributes) == bool:
                self.has_height = random.choice(self._template_attr.get("height", range(1, 12)))
                self.has_base = (
                    random.choice(self._template_attr.get("base", range(-3, 6)))
                    if pick_random()
                    else None
                )
            elif type(schematic_attributes) == dict:
                if "height" in schematic_attributes:
                    self.has_height = schematic_attributes["height"]
                if "base" in schematic_attributes:
                    self.has_base = schematic_attributes["base"]
        self._word = random.choice(["tower", "stack"])
        if repeat_key:
            self._word = make_plural(self._word)
        self.has_name = self._word


class Schematic(ComponentNode):
    """A Schematic can be either a Shape or a CategoryObject"""

    def __init__(
        self,
        only_block_type=True,  # only have a block type and no other attribute
        block_type=False,  # should the schematic have a block type
        schematic_attributes=False,  # a dict of explicit attributes requested if any
        schematic_type=None,  # type of Schematic : Shape or CategoryObject
        abstract_size=None,  # abstract size attribute
        colour=None,  # if the Schematic should have colour
        repeat_key=None,  # kind of repetition
        repeat_dir=None,  # direction of repetition
        template_attr={},
        multiple_schematics=False,
    ):
        super().__init__(template_attr=template_attr)

        self.has_size = None
        self.has_colour = None
        self._repeat = None
        self._schematics_type = None
        self.has_block_type = None

        # if the Schematic only has a block type and no other attribute
        if only_block_type:
            self.has_block_type = random.choice(
                self._template_attr.get("block_types", BLOCK_TYPES)
            )
            return

        # if the type if given assing, else pick randomly
        if schematic_type:
            schematic_type = schematic_type
        else:
            schematic_type = random.choice([Shape, CategoryObject])

        if schematic_type == Shape:  # Shape class has CHOICES
            schematic_type = random.choice(Shape.CHOICES)

        # the repeat kind. 'FOR' indicates for count, 'ALL' indicates -> for every.
        if repeat_key == "FOR":
            random_step_count = random.choice(self._template_attr.get("count", range(1, 101)))
            # pick a value for the count
            repeat_count = random.choice(
                [str(random_step_count), int_to_words(random_step_count), "a few", "some"]
            )
            self._repeat = Repeat(
                repeat_key=repeat_key, repeat_count=repeat_count, repeat_dir=repeat_dir
            )
        elif repeat_key == "ALL":
            self._repeat = Repeat(repeat_key="ALL", repeat_dir=repeat_dir)
        elif repeat_key == "ALL_ONLY":
            self._repeat = Repeat(repeat_key="ALL_ONLY")

        self._schematics_type = schematic_type(
            block_type=block_type,
            schematic_attributes=schematic_attributes,
            repeat_key=repeat_key,
            template_attr=template_attr,
        )

        # Inherit the keys from schematic type for the action tree
        for key, val in self._schematics_type.__dict__.items():
            if key.startswith("has_"):
                setattr(self, key, val)
        if multiple_schematics:
            cat_object_name = random.choice(self._template_attr.get("non_shape_names", CATEGORIES))
            shape_name = random.choice(SHAPE_NAMES)
            self.has_name = self.has_name + "__&&__" + random.choice([shape_name, cat_object_name])

        if abstract_size:  # add an abstract size if the flag is set
            self.has_size = random.choice(ABSTRACT_SIZE)
        if colour:  # add colour if colour flag is set
            self.has_colour = random.choice(COLOURS)

    def _generate_description(self):
        out_dict = OrderedDict()
        # If the Schematic only has a block_type
        if not self._schematics_type:
            out_dict["block_type"] = self.has_block_type
            return out_dict
        # get the tree from the child an expand with the parent's attributes
        child_dict = self._schematics_type.generate_description()
        if self._repeat and self._repeat.repeat_key == "FOR":
            out_dict["object_prefix"] = self._repeat.repeat_count
        if self.has_size:
            out_dict["size"] = self.has_size
        if self.has_colour:
            out_dict["colour"] = self.has_colour
        out_dict.update(child_dict)
        if "__&&__" in self.has_name:
            out_dict["word"] = self.has_name

        return out_dict


##############
## LOCATION ##
##############


class Coordinates(ComponentNode):
    """Coordinates is a list of x, y, z coordinates
    __init__(): pick the coordinates
    _generate_description() : different ways of representing coordinates in text
    """

    def __init__(self, template_attr={}):
        super().__init__(template_attr=template_attr)
        self.coordinates = random.sample(
            self._template_attr.get("coordinates", range(-100, 101)), 3
        )

    def __repr__(self):
        return "<Abs {}>".format(self.coordinates)

    def _generate_description(self):
        location = random.choice(
            [
                "at loc {} {} {}",
                "at location {} {} {}",
                "at loc: {} {} {}",
                "at location: {} {} {}",
                "at coordinates: {} {} {}",
            ]
        ).format(*self.coordinates)
        return location


class LocationDelta(ComponentNode):
    """LocationDelta picks the relative direction type.
    __init__(): pick the direction type
    """

    def __init__(
        self,
        relative_direction=True,  # indicates relative to something vs itself
        direction_type=None,  # value of direction
        additional_direction=None,
    ):
        super().__init__()
        self._relative_pos = relative_direction
        if additional_direction is None:
            additional_direction = []
        # Assign value of _direction_type to be the key name if assigned, else True
        direction_list = [
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN",
            "FRONT",
            "BACK",
            "AWAY",
        ] + additional_direction
        self._direction_type = direction_type if direction_type else random.choice(direction_list)

    def _generate_description(self):
        direction_dict = {}
        # LocationDelta being used as being relative to a BlockObject/ Mob
        # vs being used with self (AgentPos)
        if self._relative_pos:
            direction_dict["LEFT"] = ["to the left of", "towards the left of"]
            direction_dict["RIGHT"] = ["to the right of", "towards the right of"]
            direction_dict["UP"] = ["above", "on top of", "to the top of", "over the", "over"]
            direction_dict["DOWN"] = ["below", "under"]
            direction_dict["FRONT"] = ["in front of"]
            direction_dict["BACK"] = ["behind"]
            direction_dict["AWAY"] = ["away from"]
            direction_dict["INSIDE"] = ["inside"]
            direction_dict["OUTSIDE"] = ["outside"]
            direction_dict["NEAR"] = ["next to", "close to", "near"]
            direction_dict["CLOCKWISE"] = ["clockwise"]
            direction_dict["ANTICLOCKWISE"] = ["anticlockwise"]
            direction_dict["BETWEEN"] = ["between", "in between", "in the middle of"]
            direction_dict["ACROSS"] = ["across", "across from"]
        else:
            direction_dict["LEFT"] = ["to the left", "to your left", "east", "left"]
            direction_dict["RIGHT"] = ["to the right", "to your right", "right", "west"]
            direction_dict["UP"] = ["up", "north"]
            direction_dict["DOWN"] = ["down", "south"]
            direction_dict["FRONT"] = ["front", "forward", "to the front"]
            direction_dict["BACK"] = ["back", "backwards", "to the back"]
            direction_dict["AWAY"] = ["away"]
            direction_dict["CLOCKWISE"] = ["clockwise"]
            direction_dict["ANTICLOCKWISE"] = ["anticlockwise"]

        return random.choice(direction_dict[self._direction_type])


class SpeakerLook(ComponentNode):
    """SpeakerLook is where the speaker is looking.
    This class has no attributes of its own, except for the decription
    """

    def _generate_description(self):
        return random.choice(["there", "over there", "where I am looking"])


class SpeakerPos(ComponentNode):
    """SpeakerPos is where the speaker is.
    This class has no attributes of its own, except for the decription
    """

    def _generate_description(self):
        return random.choice(["here", "over here"])


class AgentPos(ComponentNode):
    """AgentPos is where the agent is.
    This class has no attributes of its own, except for the decription
    """

    def _generate_description(self):
        return random.choice(["where you are", "where you are standing"])


class Location(ComponentNode):
    """Location can be of different types: Coordinates, BlockObject, Mob,
    AgentPos, SpeakerPos, SpeakerLook
    __init__(): Pick the location_type, instantiate it (_child), pick location's coordinates
                (if location_type == "Coordinates"), instantiate relative_direction if any, pick
                number of steps.
    """

    def __init__(
        self,
        location_type=None,  # indicates whether type can be assigned and which one
        relative_direction=None,  # if relative direction id involved
        relative_direction_value=None,  # value of direction
        additional_direction=None,
        steps=False,  # does Location involves steps
        repeat_key=None,  # repeat at Location
        coref_resolve=None,  # coref resolution type
        bo_coref_resolve=None,
        template_attr={},
    ):
        def assign_relative_direction(self, relative_direction):
            if relative_direction:
                self.relative_direction = self._dirn_relative._direction_type
            elif (relative_direction is None) and (self.location_type in [BlockObject, Mob]):
                self.relative_direction = self._dirn_relative._direction_type

        super().__init__(template_attr=template_attr)
        self.steps = None
        self.relative_direction = None
        self.location_type = None
        self.contains_coreference = None

        relative_to_other = True  # if True -> relative_direction is wrt to something else
        # relative to itself.

        """Set the relative direction for location, if specified"""
        self._dirn_relative = LocationDelta(
            relative_direction=relative_to_other,
            direction_type=relative_direction_value,
            additional_direction=additional_direction,
        )
        assign_relative_direction(self, relative_direction)

        """Assign the location type"""
        # Location type None is for no location type
        if type(location_type) is not list:
            location_type = [location_type]
            bo_coref_resolve = [bo_coref_resolve] if bo_coref_resolve else None

        if self.relative_direction == "BETWEEN" and len(location_type) == 1:
            repeat_key = "ALL_ONLY"

        self._child = []
        for i, l_type in enumerate(location_type):
            if l_type == "SpeakerLookMob":
                # TODO: fix this!!
                self.location_type = SpeakerLook
            elif l_type not in [None, "ANY"]:  # specific type
                self.location_type = l_type
            elif l_type == "ANY" and relative_direction:  # relative to itself
                relative_to_other = False
                self.location_type = AgentPos
            elif coref_resolve:
                self.contains_coreference = coref_resolve
            else:
                # For type "ANY" or None specified, pick a random location
                self.location_type = random.choice([Coordinates, BlockObject, Mob])

            """Pick the child and other attributes based on location type"""
            if self.location_type not in [BlockObject, Mob]:
                location_child = self.location_type() if self.location_type else None
                self._child.append(location_child)
            if self.location_type == Coordinates:
                self.coordinates = self._child[-1].coordinates
            elif self.location_type == BlockObject:
                bo_type = None
                no_child = False
                if bo_coref_resolve and bo_coref_resolve[i]:
                    bo_type = PointedObject
                    no_child = True
                # For BlockObjects, control the depth of the tree by removing the
                # recursive location
                coreference_type = (
                    bo_coref_resolve[i] if (bo_coref_resolve and bo_coref_resolve[i]) else None
                )
                self._child.append(
                    BlockObject(
                        block_object_location=False,
                        block_object_type=bo_type,
                        repeat_key=repeat_key,
                        coref_type=coreference_type,
                        no_child=no_child,
                        template_attr=template_attr,
                    )
                )
                if len(location_type) == 1:
                    self._block_object = self._child[-1]
                elif i == 0:
                    self._block_object_1 = self._child[-1]
                elif i == 1:
                    self._block_object_2 = self._child[-1]

            # when the location_type needs to be SpeakerLook but we want properties of Mob
            # eg : "follow that pig"
            elif l_type == "SpeakerLookMob" or self.location_type == Mob:
                mob_resolve = bo_coref_resolve[i] if bo_coref_resolve else None
                self._child.append(
                    Mob(repeat_key=repeat_key, coref_type=mob_resolve, template_attr=template_attr)
                )
                if len(location_type) == 1:
                    self._mob = self._child[-1]
                elif i == 0:
                    self._mob_1 = self._child[-1]
                elif i == 1:
                    self._mob_2 = self._child[-1]
            assign_relative_direction(self, relative_direction)

        """Select steps """
        if steps:
            random_step_count = random.choice(self._template_attr.get("step", range(1, 101)))
            self.steps = random.choice(
                [str(random_step_count), int_to_words(random_step_count), "few"]
            )

    def _generate_description(self):
        out_dict = OrderedDict()  # the OrderedDict ensures that the order of values is maintained.
        if self.steps:
            out_dict["steps"] = self.steps + " steps"
        if self.relative_direction:
            out_dict["relative_direction"] = self._dirn_relative.generate_description()
        for child in self._child:
            obj_name = to_snake_case(type(child).__name__)  # key names are snake case
            if child:
                out_dict[obj_name] = child.generate_description()
        if self.contains_coreference:
            out_dict["coref"] = self.contains_coreference
        return out_dict


#################
## BLOCKOBJECT ##
#################


class Object(ComponentNode):
    """Object can be any generic thing that exists in the Minecraft environment.
    __init__(): Pick the size and colour
    The name of the object is picked in _generate_description() and can be : 'thing', 'shape', 'structure', 'object'
    """

    def __init__(
        self, repeat_key_val=None, coref_type=None, template_attr={}
    ):  # value of repeat key
        super().__init__(template_attr=template_attr)
        self._repeat = None
        self._repeat_all_text = None
        # Pick size and color at random.
        self.has_size = random.choice([""] + ABSTRACT_SIZE) if pick_random() else None
        self.has_colour = random.choice([""] + COLOURS) if pick_random() else None

        # The object can either be an abstract shape with size / color or a concrete named object.
        """If none of size and colour have been picked, deifnitely pick a name,
        to avoid : the shape , the structure etc"""
        if (not self.has_size) and (not self.has_colour):
            self.has_name = random.choice(
                self._template_attr.get("non_shape_names", CONCRETE_OBJECT_NAMES)
            )
        else:
            self.has_name = (
                random.choice(self._template_attr.get("non_shape_names", CONCRETE_OBJECT_NAMES))
                if pick_random()
                else None
            )

        # pick the kind of repetition
        if repeat_key_val:
            if repeat_key_val == "ALL":
                self._repeat = Repeat(repeat_key="ALL")
                self._repeat_all_text = random.choice(["all", "each", "every"])
            elif repeat_key_val == "ALL_ONLY":
                self._repeat = Repeat(repeat_key="ALL_ONLY")
            else:
                self._repeat = Repeat(repeat_key="FOR", repeat_count=repeat_key_val)

        # make name plural for all kind of repeats except for 'each' and 'every'.
        if self.has_name and self._repeat:
            repeat_key_type = self._repeat.repeat_key
            if (
                repeat_key_type == "FOR"
                or (repeat_key_type == "ALL" and self._repeat_all_text == "all")
                or repeat_key_type == "ALL_ONLY"
            ):
                self.has_name = make_plural(self.has_name)

    def _generate_description(self):
        out_dict = {}
        out_dict["object_prefix"] = "the"
        if self._repeat:
            repeat_key_type = self._repeat.repeat_key
            if repeat_key_type == "ALL":
                out_dict["object_prefix"] = self._repeat_all_text
            elif repeat_key_type == "FOR":
                out_dict["object_prefix"] = self._repeat.repeat_count

        if self.has_size:
            out_dict["size"] = self.has_size
        if self.has_colour:
            out_dict["colour"] = self.has_colour
        if self.has_name:
            out_dict["name"] = self.has_name

        if "name" not in out_dict:
            phrase = random.choice(["thing", "shape", "structure", "object"])
            if self._repeat:
                repeat_key_type = self._repeat.repeat_key
                if (
                    repeat_key_type == "FOR"
                    or (repeat_key_type == "ALL" and self._repeat_all_text == "all")
                    or repeat_key_type == "ALL_ONLY"
                ):
                    phrase = make_plural(phrase)
            out_dict["object"] = phrase

        return out_dict


class PointedObject(ComponentNode):
    """PointedObject is an object that the speaker is pointing at.
    __init__(): Pick the size and colour
    The name is picked later in _generate_description
    """

    KEYS = ["has_size", "has_colour"]

    def __init__(
        self, repeat_key_val=None, coref_type=None, template_attr={}
    ):  # value of repeat key
        super().__init__(template_attr=template_attr)
        self._coref_type = coref_type
        self._repeat = None
        self.has_size = random.choice([""] + ABSTRACT_SIZE) if pick_random() else None
        self.has_colour = random.choice([""] + COLOURS) if pick_random() else None
        self._word = random.choice(["this", "that"])
        # the kind of repetition
        if repeat_key_val:
            if repeat_key_val in ["ALL", "ALL_ONLY"]:
                self._repeat = Repeat(repeat_key=repeat_key_val)
            else:
                self._repeat = Repeat(repeat_key="FOR", repeat_count=repeat_key_val)

        self.has_name = (
            random.choice(self._template_attr.get("non_shape_names", CONCRETE_OBJECT_NAMES))
            if pick_random()
            else None
        )
        if self.has_name and self._repeat:
            self.has_name = make_plural(self.has_name)

    def _generate_description(self):
        out_dict = {}
        out_dict["object_prefix"] = self._word
        out_dict["object"] = None
        if self._repeat:
            repeat_key_type = self._repeat.repeat_key
            if repeat_key_type == "ALL":
                if self._coref_type:
                    out_dict["object_prefix"] = "each of"
                else:
                    out_dict["object_prefix"] = "each of " + random.choice(["those", "these"])
            elif repeat_key_type == "FOR":
                phrase = random.choice(["those ", "these "])
                if self._coref_type:
                    out_dict["object_prefix"] = self._repeat.repeat_count
                else:
                    out_dict["object_prefix"] = phrase + self._repeat.repeat_count

        if self.has_size:
            out_dict["size"] = self.has_size
        if self.has_colour:
            out_dict["colour"] = self.has_colour
        if self.has_name:
            out_dict["name"] = self.has_name

        object_description = random.choice(["thing", "shape", "structure", "object"])
        if self._repeat:
            object_description = make_plural(object_description)

        # If size or colour assigned, it needs to have a description/ name.
        if ("size" in out_dict) or ("colour" in out_dict):
            if "name" not in out_dict:
                out_dict["object"] = object_description
        else:
            if self._repeat:
                out_dict["object"] = object_description  # the plural description
            # Assign name/thing optionally
            elif "name" not in out_dict:
                out_dict["object"] = object_description if pick_random() else None

        return out_dict


class BlockObject(ComponentNode):
    """BlockObject can be anything that physically exists in the Minecraft environment.
    There can be two types of physical objects: PointedObject or an Object, as described above,
    __init__(): Pick the block_object_type, size, colour, description of the object.

    If an explicit list of block_object_attributes is given, those are assigned, otherwise
    inherit the child's attributes.
    If block_object_type is specified, assign that, else picked randomly.
    If block_object_location is specified, then assign a location else location is optional.
    """

    def __init__(
        self,
        block_object_type=None,  # the kind of BlockObject: Object / PointedObject
        block_object_attributes=None,  # explicit attributes of the blockobject
        block_object_location=None,  # the location of the blockobject if any
        no_child=False,  # should this blockobject have a child
        repeat_key=None,  # what kind of repetition
        repeat_no_child=None,  # if only repeat and no children
        repeat_location=None,  # repetition in location
        coref_type=None,  # coreference type
        template_attr={},
    ):
        super().__init__(template_attr=template_attr)
        self.has_size = None
        self.has_colour = None
        self._object_desc = None
        self._location = None
        self.has_name = None
        self._repeat = None

        block_object_repeat_cnt = None

        # the kind of repetition
        if repeat_key == "FOR":
            random_step_count = random.choice(self._template_attr.get("count", range(1, 101)))
            block_object_repeat_cnt = random.choice(
                [str(random_step_count), int_to_words(random_step_count), "few"]
            )
            self._repeat = Repeat(repeat_key=repeat_key, repeat_count=block_object_repeat_cnt)
        elif repeat_key == "ALL":
            self._repeat = Repeat(repeat_key="ALL")
            block_object_repeat_cnt = "ALL"
        elif repeat_key == "ALL_ONLY":
            self._repeat = Repeat(repeat_key="ALL_ONLY")
            block_object_repeat_cnt = "ALL_ONLY"

        # If only repeat_key and no other children
        if repeat_no_child:
            return

        """Assign block object type"""
        # If object_type is specified, assign that, else pick at random

        # for "what you built" etc
        if coref_type:
            self.contains_coreference = coref_type
        if block_object_type is None and coref_type is not None:
            return
        elif block_object_type:
            self._block_object_type = block_object_type
        else:
            self._block_object_type = Object  # PointedObject if pick_random(0.4) else Object
        self._child = self._block_object_type(
            repeat_key_val=block_object_repeat_cnt,
            coref_type=coref_type,
            template_attr=template_attr,
        )

        """Assign block object's attributes"""
        # If attribute list is not explicitly defined, pull all from
        # children if child exists.
        if not block_object_attributes and (not no_child):
            for key, val in self._child.__dict__.items():
                if key.startswith("has_"):
                    setattr(self, key, val)
        elif block_object_attributes is not None:
            # Only assign the specified attributes
            for attr in block_object_attributes:
                if attr == "size":
                    self.has_size = random.choice(ABSTRACT_SIZE)
                if attr == "colour":
                    self.has_colour = random.choice(COLOURS)
                if attr == "object":
                    thing_desc = random.choice(["thing", "shape", "structure", "object"])
                    self._object_desc = thing_desc
                if attr == "objects":
                    self._object_desc = make_plural(
                        random.choice(["thing", "shape", "structure", "object"])
                    )
                if attr == "name":
                    name_desc = random.choice(
                        self._template_attr.get("non_shape_names", CONCRETE_OBJECT_NAMES)
                    )
                    self.has_name = name_desc
                if attr == "names":
                    self.has_name = make_plural(
                        random.choice(
                            self._template_attr.get("non_shape_names", CONCRETE_OBJECT_NAMES)
                        )
                    )

        """Assign block object location"""
        if self._block_object_type == PointedObject:
            if coref_type:
                self._location = None  # Location(coref_resolve=coref_type)
            else:
                self._location = Location(location_type=SpeakerLook, template_attr=template_attr)
            # coref_type_val = coref_type if coref_type else self._child._word

        else:
            # If generic Object, if location explicitly specified then assign else optional.
            if block_object_location is True:
                self._location = Location(template_attr=template_attr)
            elif block_object_location is False:
                self._location = None
            elif block_object_location is None:
                self._location = Location(template_attr=template_attr) if pick_random() else None
            elif block_object_location in [BlockObject, Mob]:
                self._location = Location(
                    location_type=block_object_location,
                    repeat_key=repeat_location,
                    template_attr=template_attr,
                )

    def _generate_description(self):
        # Also populate things from child, in case default values are needed
        # Example when BlockObject is a reference for a Location.
        object_desc = self._child.generate_description()
        locn_desc = None
        if self._block_object_type == Object and self._location:
            locn_desc = self._location.generate_description()

        out_dict = OrderedDict()
        out_dict["object_prefix"] = object_desc["object_prefix"]
        if self.has_size:
            out_dict["size"] = self.has_size
        if self.has_colour:
            out_dict["colour"] = self.has_colour
        if self.has_name:
            out_dict["name"] = self.has_name
        if self._object_desc:
            out_dict["object"] = self._object_desc

        # We need this when BlockObject is used as a reference_object
        if "object" in object_desc and object_desc["object"]:
            out_dict["abstract_structure"] = object_desc["object"]

        if locn_desc:
            out_dict["location"] = locn_desc

        return out_dict


#################
## MOB ##
#################


class Mob(ComponentNode):
    """Mob is a mobile object in Minecraft. We have a list of these defined at the top.
    __init__(): Pick the mob name and location
    """

    def __init__(
        self,
        mob_location=None,  # specify the location of the mob if any
        repeat_key=None,  # the kind of repetition
        repeat_location=None,  # repetitions in location if allowed
        coref_type=None,
        template_attr={},
    ):
        super().__init__(template_attr=template_attr)
        self._repeat = None
        self._location = None

        self.has_name = random.choice(self._template_attr.get("mob_names", MOBS))
        self.contains_coreference = coref_type
        """Assign location of the mob if any"""
        if mob_location:
            self._location = Location(
                location_type=mob_location, repeat_key=repeat_location, template_attr=template_attr
            )

        # the kind of repetition
        if repeat_key == "FOR":
            random_step_count = random.choice(self._template_attr.get("count", range(1, 101)))
            repeat_count = random.choice(
                [str(random_step_count), int_to_words(random_step_count), "few"]
            )
            self._repeat = Repeat(repeat_key=repeat_key, repeat_count=repeat_count)
        elif repeat_key in ["ALL", "ALL_ONLY"]:
            self._repeat = Repeat(repeat_key=repeat_key)

    def _generate_description(self):
        location_desc = None
        if self._location:
            location_desc = self._location.generate_description()

        out_dict = {}
        if self._repeat:
            key_type = self._repeat.repeat_key
            if key_type == "ALL":
                out_dict["mob_prefix"] = random.choice(["all the", "each", "every", "all"])
            elif key_type == "FOR":
                out_dict["mob_prefix"] = self._repeat.repeat_count
            elif key_type == "ALL_ONLY":
                out_dict["mob_prefix"] = "the"

        elif pick_random():
            out_dict["mob_prefix"] = "the"

        # update name to be plural if repetitions
        if self._repeat:
            if ("mob_prefix" not in out_dict) or (out_dict["mob_prefix"] not in ["each", "every"]):
                self.has_name = make_plural(self.has_name)
        out_dict["mob"] = self.has_name
        if location_desc:
            out_dict["location"] = location_desc

        return out_dict


####################
## STOP CONDITION ##
####################


class StopCondition(ComponentNode):
    """Stop Condition defines the condition to terminate a loop.
    The different stop condition types are specified in : CONDITION_TYPES
    __init__(): Assign the condition type and other keys based on the condition
    """

    def __init__(
        self,
        condition_type=None,  # condition type to terminate the loop
        block_type=None,  # optional, needed for condition_type: AdjacentToBlockType
    ):
        super().__init__()
        self.condition_type = condition_type if condition_type else None
        self.block_type = (
            random.choice(self._template_attr.get("block_types", BLOCK_TYPES))
            if block_type
            else None
        )

    def _generate_description(self):
        out_dict = {}
        if self.block_type:
            out_dict["block_type"] = self.block_type
        return out_dict


##################
###   REPEAT   ###
##################


class Repeat(ComponentNode):
    """Repeat class defines the kind of loop, the number of times the loop should
    be run as well the direction of loop execution.
    """

    def __init__(
        self,
        repeat_key=None,  # the loop type: 'FOR' / 'ALL'
        repeat_count=None,  # optional, needed for counting the loop
        repeat_dir=None,  # the direction of loop execution
    ):
        super().__init__()
        self.repeat_key = repeat_key if repeat_key else None
        self.repeat_count = repeat_count if repeat_count else None
        self.repeat_dir = repeat_dir if repeat_dir else None

    def _generate_description(self):
        return {}


###################
###   FILTERS   ###
###################


class Filters(ComponentNode):
    """Filters class defines the name of filters and their values.
    This is used with GetMemory and PutMemory actions
    """

    def __init__(
        self,
        has_tag=None,  # the has_tag value
        mem_type=None,  # type of task
        has_name=None,
        block_object_attr=None,
        bo_updated=False,
        location_attr=None,
        location_updated=False,
        mob_attr=None,
        mob_updated=False,
    ):
        super().__init__()
        self.has_tag = has_tag if has_tag else None
        self.type = mem_type if mem_type else None
        self.has_name = has_name if has_name else None
        self._location = None
        self._reference_object = None

        if bo_updated:
            self._reference_object = BlockObject(**block_object_attr)
        if location_updated:
            self._location = Location(**location_attr)
        if mob_updated:
            self._reference_object = Mob(**mob_attr)

    def _generate_description(self):
        out_dict = {}
        if self._reference_object:
            out_dict = self._reference_object.generate_description()
        if self._location:
            out_dict.update(self._location.generate_description())

        return out_dict


###################
###   UPSERT   ###
###################
class Upsert(ComponentNode):
    """Upsert class defines the memoryt type and data that needs to be upserted
    into the bot's memory
    """

    def __init__(
        self, memory_type=None, reward_value=None, has_tag=None, has_size=None, has_colour=None
    ):
        super().__init__()
        self._memory_data = None
        if memory_type:
            self._memory_data = MemoryData(
                reward_value=reward_value,
                memory_type=memory_type,
                has_tag=has_tag,
                has_size=has_size,
                has_colour=has_colour,
            )

    def _generate_description(self):
        return {}


######################
###   MemoryData   ###
######################
class MemoryData(ComponentNode):
    def __init__(
        self, reward_value=None, memory_type=None, has_tag=None, has_colour=None, has_size=None
    ):
        super().__init__()
        self.reward_value = reward_value if reward_value else None
        self.memory_type = memory_type if memory_type else None
        self.has_tag = has_tag if has_tag else None
        self.has_size = has_size if has_size else None
        self.has_colour = has_colour if has_colour else None

    def _generate_description(self):
        return {}


##################
## CHOICE LISTS ##
##################

Shape.CHOICES = [
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
    BlockShape,
]
