"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with repeats and stop
conditions.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *

################################
### STOP CONDITION TEMPLATES ###
################################
"""Assign the repeat_val to the tree"""


def assign_repeat_key(template_obj, repeat_val, templ_index=0):
    template_names = get_template_names(template_obj, templ_index)

    if any(x in ["Build", "BuildSingle", "Stack", "Place"] for x in template_names):
        template_obj.node._location_args["repeat_key"] = repeat_val
    elif any(x in ["Dig", "Fill"] for x in template_names):
        template_obj.node._location_args["repeat_key"] = repeat_val
    elif any(x in ["Destroy", "Freebuild"] for x in template_names):
        template_obj.node._block_obj_args["block_object_location"] = random.choice(
            [BlockObject, Mob]
        )
        template_obj.node._block_obj_args["repeat_location"] = repeat_val
    elif "Tag" in template_names:
        if "MobName" in template_names:
            template_obj.node._mob_args["mob_location"] = random.choice([BlockObject, Mob])
            template_obj.node._mob_args["repeat_location"] = repeat_val
        else:
            template_obj.node._block_obj_args["block_object_location"] = random.choice(
                [BlockObject, Mob]
            )
            template_obj.node._block_obj_args["repeat_location"] = repeat_val


class RepeatAllLocation(TemplateObject):
    """This template object repesents all / every / each for blockobject / mob used
    as a reference for location.
    eg: explode the disk on top of all the creepers"""

    def add_generate_args(self, index=0, templ_index=0):
        assign_repeat_key(self, "ALL", templ_index=templ_index)


class RepeatCountLocation(TemplateObject):
    """This template object repesents a count for blockobject / mob used
    as a reference for location.
    eg: Make a hole below 5 houses"""

    def add_generate_args(self, index=0, templ_index=0):
        assign_repeat_key(self, "FOR", templ_index=templ_index)


class RepeatCount(TemplateObject):
    """This template object repesents a count for blockobject / mob / schematic
    eg: Make 5 holes there"""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if any(x in ["Build", "Stack", "Place"] for x in template_names):
            self.node._schematics_args["repeat_key"] = "FOR"
        elif any(x in ["Dig", "Fill"] for x in template_names):
            count = random.choice(self.template_attr.get("count", range(2, 15)))
            self.node._repeat_args["repeat_key"] = "FOR"
            self.node._repeat_args["repeat_count"] = random.choice(
                [str(count), int_to_words(count), "a few", "some"]
            )
        elif any(x in ["Destroy", "Freebuild"] for x in template_names):
            self.node._block_obj_args["repeat_key"] = "FOR"
        elif "Copy" in template_names:
            self.node._block_obj_args["repeat_key"] = "FOR"
            self.node._block_obj_args["no_child"] = True
        elif "Tag" in template_names:
            if "MobName" in template_names:
                self.node._mob_args["repeat_key"] = "FOR"
            else:
                self.node._block_obj_args["repeat_key"] = "FOR"
                self.node._block_obj_args["no_child"] = True
        elif "Spawn" in template_names:
            self.node._mob_args["repeat_key"] = "FOR"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        if len(self.node.args) > 0:
            description = self.node.args[arg_index].generate_description()
            if "object_prefix" in description:
                return description["object_prefix"]
        template_names = get_template_names(self, templ_index)
        if ("Dig" in template_names) or ("Fill" in template_names):
            return self.node._repeat_args["repeat_count"]

        return None


class Around(TemplateObject):
    """This TemplateObject populates the repeat_dir as "Around" """

    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["repeat_dir"] = "AROUND"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "around"


class Surround(TemplateObject):
    """This TemplateObject populates the repeat_dir as "SURROUND" """

    def add_generate_args(self, index=0, templ_index=0):
        self.node._schematics_args["repeat_key"] = "ALL"
        self.node._schematics_args["repeat_dir"] = "SURROUND"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "surround"


class RepeatAll(TemplateObject):
    """This template object repesents "All" for blockobject / mob / schematic
    eg: Tag every house as brown"""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        template_len = len(template_names) - 1
        # only for destroy everything.
        if (template_len == 2) and ("DestroySingle" in template_names):
            self.node._block_obj_args["repeat_no_child"] = True

        if "Build" in template_names:
            self.node._schematics_args["repeat_key"] = "ALL"
        elif any(x in ["Destroy", "Freebuild", "DestroySingle"] for x in template_names):
            self.node._block_obj_args["repeat_key"] = "ALL"
        elif "Copy" in template_names:
            self.node._block_obj_args["repeat_key"] = "ALL"
            self.node._block_obj_args["no_child"] = True
        elif "Fill" in template_names:
            self.node._repeat_args["repeat_key"] = "ALL"
        elif "Tag" in template_names:
            if "MobName" in template_names:
                self.node._mob_args["repeat_key"] = "ALL"
            else:
                self.node._block_obj_args["repeat_key"] = "ALL"
                self.node._block_obj_args["no_child"] = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if ("Copy" in template_names) and (len(self.node.args) > 0):
            description = self.node.args[arg_index].generate_description()
            if "object_prefix" in description:
                return description["object_prefix"]
        if "Fill" in template_names:
            return random.choice(["all", "all the"])
        return None


class ConditionTypeNever(TemplateObject):
    """This template object repesents the condition type "Never" for the Stop Condition
    i.e. infinite loops.
    eg: Follow me"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._condition_args["condition_type"] = "NEVER"


class ConditionTypeAdjacentBlockType(TemplateObject):
    """This template object repesents the condition type "adjacent to block type"
    for Stop Condition i.e. stop when you are adjacent to a certain block type.
    eg: Dig until you hit bedrock"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._condition_args["condition_type"] = "ADJACENT_TO_BLOCK_TYPE"
        self.node._condition_args["block_type"] = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        block_type = description["block_type"]
        template_names = get_template_names(self, templ_index)

        prefixes = ["until you see", "until you reach", "until you find", "down to", "for"]
        if "DownTo" in template_names:
            prefixes = [""]
        elif template_names[1] == "Dig":  # some Move templates also have this
            prefixes.append("until you hit")

        prefix = random.choice(prefixes)
        phrase = " ".join([prefix, block_type]).strip()
        return phrase


CONDIITON_TEMPLATES = [ConditionTypeNever, ConditionTypeAdjacentBlockType]
REPEAT_KEY_TEMPLATES = [RepeatCount, RepeatAll]
