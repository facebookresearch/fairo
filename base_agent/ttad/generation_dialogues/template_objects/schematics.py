"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with Schematics.
"""
import random

from ..generate_utils import *
from ..tree_components import *
from .template_object import *

###########################
### SCHEMATIC TEMPLATES ###
###########################


class UsingBlockType(TemplateObject):
    """This template ensures that the final generation has an attribute : block_type."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if any(x in ["BuildSingle", "Use"] for x in template_names):
            self.node._schematics_args["only_block_type"] = True
            if (
                "Use" in template_names
                and self.node._schematics_args["schematic_type"] is not None
            ):
                self.node._schematics_args["only_block_type"] = False

        self.node._schematics_args["block_type"] = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        block_type = description["block_type"]
        template_name = get_template_names(self, templ_index)

        previous_template_name = template_name[index - 1]
        if previous_template_name == "Build":
            return prepend_a_an(block_type)
        return block_type


class AndBuild(TemplateObject):
    """This TemplateObject represents that there are two schematics that need to be built"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["multiple_schematics"] = True


class DescribingWord(TemplateObject):
    """This template object repesents the word / name of a schematic."""

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["word"]

        template_names = get_template_names(self, templ_index)
        previous_template_name = template_names[index - 1]
        # with a 0.5 probability, return with a/an
        if previous_template_name == "Build":
            if pick_random():
                return prepend_a_an(word)

        return word


class NumBlocks(TemplateObject):
    """This template object represents a number of blocks and can be used to assign
    height, width, depth etc."""

    def add_generate_args(self, index=0, templ_index=0):
        height = random.choice(self.template_attr.get("height", range(1, 12)))
        self.num = int_to_words(height) if pick_random() else height

        action_node = self.node
        previous_template_name = type(action_node.template[templ_index][1]).__name__

        if previous_template_name == "Dig":
            dim_template = type(action_node.template[templ_index][index + 2]).__name__
            if dim_template == "Wide":
                action_node.schematic["has_width"] = self.num
            elif dim_template == "Long":
                action_node.schematic["has_length"] = self.num
            elif dim_template == "Deep":
                action_node.schematic["has_depth"] = self.num
        elif previous_template_name == "Build":
            dim_template = type(action_node.template[templ_index][index + 2]).__name__
            node_attr = self.node._schematics_args["schematic_attributes"]

            if dim_template == "High":
                if node_attr:
                    node_attr["height"] = self.num
                else:
                    self.node._schematics_args["schematic_attributes"] = {"height": self.num}
            if dim_template == "Long":
                if node_attr:
                    node_attr["length"] = self.num
                else:
                    self.node._schematics_args["schematic_attributes"] = {"length": self.num}

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if template_names[index - 1] == "Build":
            return "a " + str(self.num)

        return str(self.num)


"""This template object represents an A by B dimension of a Schematic.
Eg: The 2 x 4 in "Build a 2 x 4 wall" """


class SchematicsDimensions(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["schematic_attributes"] = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        template_names = get_template_names(self, templ_index)
        dimensions = None
        multi_size_attr = None

        # shape_attributes isn't supported for CategoryObject Schematics
        # extract dimensions from the dict
        if "shape_attributes" in description:
            shape_attributes = description["shape_attributes"]
            for attr in shape_attributes:
                if "size" in attr:
                    size = attr.split("size")[1].strip()  # extract '3 x 4' from 'of size 3 x 4'
                    if ("x" in size) or ("by" in size):
                        # if already formatted
                        multi_size_attr = attr
                        dimensions = size
                    else:
                        # else construct dimensions
                        sizes = size.split()
                        if len(sizes) > 1:
                            sizes = random.choice([" x ".join(sizes), " by ".join(sizes)])
                            multi_size_attr = attr
                            dimensions = sizes

            if multi_size_attr:
                shape_attributes.remove(multi_size_attr)

            if dimensions and ("RepeatCount" not in template_names):
                dimensions = "a " + dimensions  # a 3 x 4 cube

            return dimensions
        return ""


"""This template object forces the Schematics to have explicit
attributes / dimensions"""


class WithAttributes(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["schematic_attributes"] = True

    def generate_description(self, arg_index, index, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        # shape_attributes isn't supported for CategoryObject Schematics
        if "shape_attributes" in description:
            shape_attributes = description["shape_attributes"]
            return shape_attributes
        return ""


"""This template object adds an abstract 'size' to Schematics"""


class SchematicSize(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["abstract_size"] = True

    def generate_description(self, arg_index, index, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        size = description["size"]

        template = get_template_names(self, templ_index)
        previous_template_name = template[index - 1]
        # For Build: Build a huge archway / Build huge archway
        if previous_template_name == "Build":
            if pick_random():
                return prepend_a_an(size)
        return size


"""This template object adds an abstract 'color' to Schematics"""


class SchematicColour(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["colour"] = True

    def generate_description(self, arg_index, index, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        colour = description["colour"]

        template = get_template_names(self, templ_index)
        previous_template_name = template[index - 1]
        # For Build: Build a red archway / Build red archway
        if previous_template_name == "Build":
            if pick_random():
                return prepend_a_an(colour)
        return colour


"""This template object represents one word shapes for Build commands.
'dome' -> Build dome"""


class BuildShape(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self._shape_type = random.choice(SCHEMATIC_TYPES)
        self.node._schematics_args["schematic_type"] = self._shape_type

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["word"]

        return word


class Use(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._no_children = False

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["use", "make using", "build using", "do the construction using"])
        return phrase


SCHEMATICS_TEMPLATES = [
    UsingBlockType,
    DescribingWord,
    SchematicsDimensions,
    WithAttributes,
    NumBlocks,
    SchematicSize,
    SchematicColour,
    BuildShape,
    Use,
]
