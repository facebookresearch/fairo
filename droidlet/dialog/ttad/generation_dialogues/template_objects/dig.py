"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with Dig action.
"""
import random

from ..generate_utils import *
from ..tree_components import *
from .template_object import *

#####################
### DIG TEMPLATES ###
#####################
dig_shapes = ["hole", "cave", "mine", "tunnel"]


"""This template object picks the shape of what will be dug"""


class DigSomeShape(TemplateObject):
    def __init__(self, node, template_attr):
        shape_type = DigShapeAny if pick_random(0.8) else DigShapeHole
        self._child = shape_type(node=node, template_attr=template_attr)

    def add_generate_args(self, index=0, templ_index=0):
        self._child.add_generate_args(index=index, templ_index=templ_index)

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._child.generate_description(arg_index=arg_index, index=index)


"""This template object represents specific shapes. Meant to generate direct
commands like : make a hole , dig a mine"""


class DigShapeHole(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self._phrase = None
        this_phrase = None

        template_name = get_template_names(self, templ_index)
        plural = False
        phrase = random.choice(dig_shapes)
        this_phrase = phrase
        if "RepeatCount" in template_name:
            phrase = make_plural(random.choice(dig_shapes))
            this_phrase = phrase
            plural = True
        if not plural and (template_name[index - 1] not in ["DigDimensions", "DigAbstractSize"]):
            this_phrase = phrase
            phrase = random.choice([phrase, prepend_a_an(phrase)])
        self.node.schematic["has_name"] = this_phrase
        self._phrase = phrase

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._phrase


"""This template object covers a variety of dig shape types and is more general
than DigShapeHole. It can also lead to generations like: 'dig down until you hit bedrock'
"""


class DigShapeAny(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self._phrase = None
        template_name = get_template_names(self, templ_index=0)
        this_phrase = None
        if "RepeatCount" in template_name:
            phrase = make_plural(random.choice(dig_shapes))
            this_phrase = phrase
        elif "DownTo" in template_name:
            if pick_random():
                this_phrase = random.choice(dig_shapes + ["grass"])
                phrase = this_phrase
            else:
                this_phrase = random.choice(dig_shapes)
                phrase = "a " + this_phrase
        elif template_name[index - 1] in ["DigDimensions", "DigAbstractSize"]:
            phrase = random.choice(dig_shapes)
            this_phrase = phrase
        elif index + 1 < len(template_name) and template_name[index + 1] == "NumBlocks":
            if pick_random():
                this_phrase = random.choice(dig_shapes + ["under ground", "grass"])
                phrase = this_phrase
            else:
                this_phrase = random.choice(dig_shapes)
                phrase = "a " + this_phrase
        else:
            if pick_random():
                this_phrase = random.choice(
                    dig_shapes
                    + ["ground", "into ground", "under ground", "under grass", "grass", "down"]
                )
                phrase = this_phrase
            else:
                this_phrase = random.choice(dig_shapes)
                phrase = "a " + this_phrase
        self.node.schematic["has_name"] = this_phrase
        self._phrase = phrase

    def generate_description(self, arg_index=0, index=0, previous_text=None, templ_index=0):
        return self._phrase


"""This template object assigns the dimensions: length, width and depth for
what needs to be dug."""


class DigDimensions(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node.schematic["has_length"] = random.choice(
            self.template_attr.get("length", range(2, 15))
        )
        width_val = None
        if pick_random():
            width_val = random.choice(self.template_attr.get("width", range(15, 30)))
        if width_val:
            self.node.schematic["has_width"] = width_val

        depth_val = None
        if pick_random():
            depth_val = random.choice(self.template_attr.get("width", range(30, 45)))
        if depth_val:
            self.node.schematic["has_depth"] = depth_val

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_name = get_template_names(self, templ_index)
        sizes = [self.node.schematic["has_length"]]

        if "has_width" in self.node.schematic:
            sizes.append(self.node.schematic["has_width"])
        if "has_depth" in self.node.schematic:
            sizes.append(self.node.schematic["has_depth"])
        out_size = random.choice([" x ".join(map(str, sizes)), " by ".join(map(str, sizes))])

        if ("RepeatCount" in template_name) or ("OfDimensions" in template_name):
            return out_size

        return "a " + out_size


"""This template object assigns an abstract size for the shape that needs
to be dug."""


class DigAbstractSize(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self._size_description = random.choice(
            ABSTRACT_SIZE + ["deep", "very deep", "really deep"]
        )
        self.node.schematic["has_size"] = self._size_description

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if "RepeatCount" in template_names:
            return self._size_description
        phrase = random.choice([self._size_description, prepend_a_an(self._size_description)])
        return phrase


DIG_SHAPE_TEMPLATES = [DigSomeShape, DigShapeHole, DigShapeAny, DigDimensions, DigAbstractSize]
