"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with BlockObjects.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *
from .location import *


#############################
### BLOCKOBJECT TEMPLATES ###
#############################

# TODO: refactor this function.
def define_block_object_type(
    template_obj,
    template_obj_name,
    index,
    reference_type=None,
    block_obj_name=PointedObject,
    templ_index=0,
):
    template = get_template_names(template_obj, templ_index)

    previous_template = None
    if templ_index - 1 >= 0:
        previous_template = get_template_names(template_obj, templ_index - 1)

    if (
        (
            any(
                x
                in [
                    "Dig",
                    "Fill",
                    "FreebuildLocation",
                    "Copy",
                    "Build",
                    "BuildSingle",
                    "Stack",
                    "Place",
                    "Surround",
                ]
                for x in template
            )
            or (
                previous_template
                and any(
                    x
                    in [
                        "Fill",
                        "FreebuildLocation",
                        "Copy",
                        "Build",
                        "BuildSingle",
                        "Stack",
                        "Place",
                        "Surround",
                    ]
                    for x in previous_template
                )
            )
        )
        and (
            (index - 1 >= 0)
            and (
                template[index - 1]
                in ["RelativeDirectionTemplate", "Around", "Surround", "Between"]
            )
        )
    ) or (any(x in ["Move", "Stand", "Down"] for x in template)):
        if (
            "Between" in template
            and type(template_obj.node._location_args["location_type"]) is list
        ):
            template_obj.node._location_args["location_type"].append(BlockObject)
            template_obj.node._location_args["bo_coref_resolve"].append(reference_type)
        else:
            template_obj.node._location_args["location_type"] = [BlockObject]
            template_obj.node._location_args["bo_coref_resolve"] = [reference_type]
    else:
        template_obj.node._block_obj_args["block_object_type"] = block_obj_name  # at SpeakerLook
        if reference_type:
            template_obj.node._block_obj_args["coref_type"] = reference_type

        # If this is the last template object, no children
        # If followed by is/looks like, no child
        # If followed by a location, no child

        location_template_names = [loc.__name__ for loc in LOCATION_TEMPLATES]
        if (
            (template[-1] == template_obj_name)
            or (
                any(
                    x
                    in [
                        "BlockObjectThat",
                        "BlockObjectCoref",
                        "BlockObjectThese",
                        "BlockObjectThose",
                        "BlockObjectThis",
                        "BlockObjectIt",
                    ]
                    for x in template
                )
            )
            or (template[index + 1] in ["NTimes", "ForMe", "TagDesc", "TagName"])
            or (
                (
                    index + 1 < len(template)
                    and (template[index + 1] in ["Is", "Looks", "With"] + location_template_names)
                )
            )
        ):
            template_obj.node._block_obj_args["no_child"] = True


class BlockObjectCoref(TemplateObject):
    """This template object represents a BlockObject where the Speaker is pointing.
    ("this")"""

    def add_generate_args(self, index=0, templ_index=0):
        self._words = random.choice(
            [
                "what you built",
                "what you just built",
                "what you built just now",
                "what you made",
                "what you constructed just now",
            ]
        )
        define_block_object_type(
            self,
            "BlockObject",
            index,
            reference_type="yes",  # self._words,
            block_obj_name=None,
            templ_index=templ_index,
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._words


class BlockObjectThis(TemplateObject):
    """This template object represents a BlockObject that can be a coreference.
    ("this")"""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        self._coref_val = "yes"  # "this"
        self._word = "this"
        if any(x in ["RepeatAll", "RepeatCount"] for x in template_names):
            self._word = random.choice(["these", "those"])

        define_block_object_type(
            self, "BlockObjectThis", index, reference_type=self._coref_val, templ_index=templ_index
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._word


class BlockObjectThese(TemplateObject):
    """This template object represents a BlockObject that can be a coreference
    ("these")"""

    def add_generate_args(self, index=0, templ_index=0):
        # "these"
        define_block_object_type(
            self, "BlockObjectThese", index, reference_type="yes", templ_index=templ_index
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "these"


class BlockObjectThat(TemplateObject):
    """This template object represents a BlockObject that can be a coreference
    ("that")"""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        self._word = "that"

        if any(x in ["RepeatCountLocation", "RepeatAllLocation"] for x in template_names):
            self._word = random.choice(["these", "those"])
        elif index + 1 < len(template_names) and template_names[index + 1] == "RepeatCount":
            self._word = random.choice(["these", "those"])

        self._coref_val = "yes"  # self._word

        define_block_object_type(
            self, "BlockObjectThat", index, reference_type=self._coref_val, templ_index=templ_index
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        template_names = get_template_names(self, templ_index)

        if index + 1 < len(template_names):
            if template_names[index + 1] == "RepeatAllLocation":
                self._word = " ".join([random.choice(["each of", "all of"]), self._word])
            elif template_names[index + 1] == "RepeatCountLocation":
                if "object_prefix" in description:
                    count = description["object_prefix"]
                elif (
                    "block_object" in description
                    and "object_prefix" in description["block_object"]
                ):
                    count = description["block_object"]["object_prefix"]
                self._word = " ".join([count, "of", self._word])

        return self._word


class BlockObjectIt(TemplateObject):
    """This template object represents a BlockObject that can be a coreference.
    ("it")"""

    def add_generate_args(self, index=0, templ_index=0):
        # "it"
        define_block_object_type(
            self, "BlockObjectIt", index, reference_type="yes", templ_index=templ_index
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "it"


class BlockObjectThose(TemplateObject):
    """This template object represents a BlockObject that can be a coreference.
    ("those")"""

    def add_generate_args(self, index=0, templ_index=0):
        # "those"
        define_block_object_type(
            self, "BlockObjectThose", index, reference_type="yes", templ_index=templ_index
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "those"


class AbstractDescription(TemplateObject):
    """This template object represents abstract description of a block object
    like: 'object', 'thing' etc"""

    def add_generate_args(self, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        if templ_index - 1 >= 0:
            prev_template_names = get_template_names(self, templ_index - 1)

        if ("RepeatCount" in all_names) or ("All" in all_names):
            # plural
            if self.node._block_obj_args["block_object_attributes"]:
                self.node._block_obj_args["block_object_attributes"].append("objects")
            else:
                self.node._block_obj_args["block_object_attributes"] = ["objects"]
        else:
            if self.node._block_obj_args["block_object_attributes"]:
                self.node._block_obj_args["block_object_attributes"].append("object")
            else:
                self.node._block_obj_args["block_object_attributes"] = ["object"]

        if "HumanReplace" in all_names:
            if any(
                x
                in [
                    "BlockObjectThis",
                    "BlockObjectThat",
                    "BlockObjectIt",
                    "BlockObjectThose",
                    "BlockObjectThese",
                ]
                for x in prev_template_names
            ):
                # unset the coref resolve if it was set in previous command
                if self.node._block_obj_args["coref_type"]:
                    self.node._block_obj_args["coref_type"] = None

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["object"]

        prev_template_names = None
        if templ_index - 1 >= 0:
            prev_template_names = get_template_names(self, templ_index - 1)
        if prev_template_names and "RepeatCount" in prev_template_names:
            word = make_plural(word)

        return word


class ConcreteDescription(TemplateObject):
    """This template object represents concrete description / name of a block object"""

    def add_generate_args(self, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        if templ_index - 1 >= 0:
            prev_template_names = get_template_names(self, templ_index - 1)

        if ("RepeatCount" in all_names) or ("All" in all_names):
            # plural
            if self.node._block_obj_args["block_object_attributes"]:
                self.node._block_obj_args["block_object_attributes"].append("names")
            else:
                self.node._block_obj_args["block_object_attributes"] = ["names"]
        else:
            if self.node._block_obj_args["block_object_attributes"]:
                self.node._block_obj_args["block_object_attributes"].append("name")
            else:
                self.node._block_obj_args["block_object_attributes"] = ["name"]

        if "HumanReplace" in all_names:
            if any(
                x
                in [
                    "BlockObjectThis",
                    "BlockObjectThat",
                    "BlockObjectIt",
                    "BlockObjectThose",
                    "BlockObjectThese",
                ]
                for x in prev_template_names
            ):
                # unset the coref resolve if it was set in previous command
                if self.node._block_obj_args["coref_type"]:
                    self.node._block_obj_args["coref_type"] = None

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["name"]
        node_template = get_template_names(self, templ_index)
        if node_template[index - 1] == "Copy":
            word = random.choice([word, prepend_a_an(word)])
        if "AskIs" in node_template and index == len(node_template) - 1:
            word = prepend_a_an(word)
        return word


class Colour(TemplateObject):
    """This template object ensures that the blockobject has a colour and is used
    to separate out the attributes, one at a time."""

    def add_generate_args(self, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        if templ_index - 1 >= 0:
            prev_template_names = get_template_names(self, templ_index - 1)

        if "HumanReplace" in all_names:
            if any(
                x
                in [
                    "BlockObjectThis",
                    "BlockObjectThat",
                    "BlockObjectIt",
                    "BlockObjectThose",
                    "BlockObjectThese",
                ]
                for x in prev_template_names
            ):
                # unset the coref resolve if it was set in previous command
                if self.node._block_obj_args["coref_type"]:
                    self.node._block_obj_args["coref_type"] = None

        self.node._block_obj_args["block_object_type"] = Object
        if self.node._block_obj_args["block_object_attributes"]:
            self.node._block_obj_args["block_object_attributes"].append("colour")
        else:
            self.node._block_obj_args["block_object_attributes"] = ["colour"]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["colour"]
        return word


class Size(TemplateObject):
    """This template object ensures that the blockobject has a size and is used
    to separate out the attributes, one at a time."""

    def add_generate_args(self, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        if templ_index - 1 >= 0:
            prev_template_names = get_template_names(self, templ_index - 1)

        if "HumanReplace" in all_names:
            if any(
                x
                in [
                    "BlockObjectThis",
                    "BlockObjectThat",
                    "BlockObjectIt",
                    "BlockObjectThose",
                    "BlockObjectThese",
                ]
                for x in prev_template_names
            ):
                # unset the coref resolve if it was set in previous command
                if self.node._block_obj_args["coref_type"]:
                    self.node._block_obj_args["coref_type"] = None

        self.node._block_obj_args["block_object_type"] = Object
        if self.node._block_obj_args["block_object_attributes"]:
            self.node._block_obj_args["block_object_attributes"].append("size")
        else:
            self.node._block_obj_args["block_object_attributes"] = ["size"]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["size"]
        return word


class BlockObjectLocation(TemplateObject):
    """This template object ensures that the blockobject has a location."""

    def add_generate_args(self, index=0, templ_index=0):
        curr_template_names = get_template_names(self, templ_index)
        if templ_index - 1 >= 0:
            prev_template_names = get_template_names(self, templ_index - 1)

        # For Answer action, make child None and location specific
        if curr_template_names[1] in ["What", "WhatSee", "AskIs", "AskSize"]:
            self.node._block_obj_args["block_object_location"] = random.choice([BlockObject, Mob])
            self.node._block_obj_args["no_child"] = True
        elif "HumanReplace" in curr_template_names:
            if any(
                x
                in [
                    "What",
                    "AskSize",
                    "AskColour",
                    "AskIs",
                    "Copy",
                    "Destroy",
                    "TagDesc",
                    "TagName",
                ]
                for x in prev_template_names
            ):
                self.node._block_obj_args["block_object_location"] = random.choice(
                    [BlockObject, Mob]
                )
                self.node._block_obj_args["no_child"] = True
            # if BO type was set to Pointed Object, reset it
            if any(
                x
                in [
                    "BlockObjectThis",
                    "BlockObjectThat",
                    "BlockObjectIt",
                    "BlockObjectThose",
                    "BlockObjectThese",
                ]
                for x in prev_template_names
            ):
                # unset the coref resolve if it was set in previous command
                if self.node._block_obj_args["coref_type"]:
                    self.node._block_obj_args["coref_type"] = None
                self.node._block_obj_args["block_object_type"] = Object

        else:
            # pick any random location
            self.node._block_obj_args["block_object_location"] = True

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["location"]
        return word


BLOCKOBJECT_TEMPLATES = [
    BlockObjectIt,
    AbstractDescription,
    ConcreteDescription,
    Colour,
    Size,
    BlockObjectLocation,
    BlockObjectThat,
    BlockObjectThis,
    BlockObjectThese,
    BlockObjectCoref,
]
