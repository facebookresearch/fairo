"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with special shape commands
like : Wall, Stack, Place etc
"""
from droidlet.base_util import prepend_a_an
from .template_object import *

##################################
## SPECIAL COMMANDS FOR SHAPES ###
##################################


class Wall(TemplateObject):
    """These template objects represent Schematics of type Shape that have an additional
    'has_shape_' key in their dictionary.
    This is mostly because the description of these doesn't occur in the
    surface form to compute spans."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["schematic_type"] = RectangleShape

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


class Stack(TemplateObject):
    """Rectanguloid with a height n"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["schematic_type"] = Shape
        self.node._schematics_args["repeat_dir"] = "UP"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(["stack", "put up"])
        prefix = random.choice(
            ["", random.choice(["can you", "please", "can you please", "let 's", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()
        return new_command


class Place(TemplateObject):
    """Rectangualoid with a width n"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["schematic_type"] = Shape
        self.node._schematics_args["repeat_dir"] = "RIGHT"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(["place", "put"])
        prefix = random.choice(
            ["", random.choice(["can you", "please", "can you please", "let 's", "help me"])]
        )
        new_command = (" ".join([prefix, command])).strip()
        return new_command
