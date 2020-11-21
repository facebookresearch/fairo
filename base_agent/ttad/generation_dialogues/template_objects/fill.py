"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with Fill action.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *
from .dig import *

#####################
## FILL TEMPLATES ###
#####################


class FillShape(TemplateObject):
    """This template object repesents the shape/ thing that needs to be filled."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        phrase = random.choice(dig_shapes)
        if any(x in ["RepeatCount", "RepeatAll"] for x in template_names):
            phrase = make_plural(random.choice(dig_shapes))

        self.node.reference_object["has_name"] = phrase
        self._phrase = phrase

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._phrase


# Note: this is for "fill that mine" , no coref resolution needed
class FillObjectThis(TemplateObject):
    """This template object repesents that the thing to be filled is where the speaker
    is looking."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node.reference_object["contains_coreference"] = "yes"
        phrases = ["this", "that"]
        template_names = get_template_names(self, templ_index)

        if any(x in ["RepeatCount", "RepeatAll"] for x in template_names):
            phrases = ["these", "those"]

        self._word = random.choice(phrases)

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._word


class FillBlockType(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node.has_block_type = random.choice(
            self.template_attr.get("block_types", BLOCK_TYPES)
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.node.has_block_type


class UseFill(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["use", "fill using", "fill with"])
        return phrase
