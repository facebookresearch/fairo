"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains common template objects used across different templates.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *

#######################
## DANCE TEMPLATES  ##
#######################

"""
Fly
Jump
Hop
Dance
Dance clockwise (#4 with repeat_dir)
Dance in a loop
"""


class Fly(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        command = ["fly", "fly"]
        if index + 1 < len(all_names) and all_names[index + 1] == "ConditionTypeNever":
            command = random.choice(
                [["flying", "keep flying"], ["fly", "fly until I tell you to stop"]]
            )

        self.node.dance_type_name = command[0]
        self.node._dance_text = command[1]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = self.node._dance_text
        prefix = random.choice(["", random.choice(["", "can you", "please", "can you please"])])
        new_command = (" ".join([prefix, command])).strip()

        return new_command


class Jump(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        command = ["jump", "jump"]
        if index + 1 < len(all_names) and all_names[index + 1] == "ConditionTypeNever":
            command = random.choice(
                [["jumping", "keep jumping"], ["jump", "jump until I tell you to stop"]]
            )

        self._dance_text = command[1]
        self.node.dance_type_name = command[0]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = self._dance_text
        prefix = random.choice(["", random.choice(["", "can you", "please", "can you please"])])
        new_command = " ".join([prefix, command]).strip()

        return new_command


class Walk(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node.dance_pattern = "MOVE_AROUND"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(["move", "walk", "go"])
        prefix = random.choice(["", random.choice(["", "can you", "please", "can you please"])])
        new_command = (" ".join([prefix, command])).strip()

        return new_command


class Hop(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        command = ["hop", "hop"]
        if index + 1 < len(all_names) and all_names[index + 1] == "ConditionTypeNever":
            command = random.choice(
                [["hopping", "keep hopping"], ["hop", "hop until I tell you to stop"]]
            )

        self.node.dance_type_name = command[0]
        self.node._dance_text = command[1]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = self.node._dance_text
        prefix = random.choice(["", random.choice(["", "can you", "please", "can you please"])])
        new_command = (" ".join([prefix, command])).strip()

        return new_command
