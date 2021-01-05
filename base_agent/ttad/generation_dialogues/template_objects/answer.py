"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated specifically with the
Answer action.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *


########################
### ANSWER TEMPLATES ###
########################


class What(TemplateObject):
    '''This template object represents questions of type "what .."'''

    def add_generate_args(self, index=0, templ_index=0):
        self.node.answer_type = "TAG"
        self.node.tag_name = "NAME"
        self.node._filters_args["mem_type"] = "REFERENCE_OBJECT"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["what", "what do you think"])

        return phrase


class WhatSee(TemplateObject):
    '''This template object represents questions of type: "what do you see at.."'''

    def add_generate_args(self, index=0, templ_index=0):
        self.node.answer_type = "TAG"
        self.node.tag_name = "NAME"
        self.node._filters_args["mem_type"] = "REFERENCE_OBJECT"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["what do you see", "what do you observe"])

        return phrase


class AskSize(TemplateObject):
    '''This template object repesents questions of type: "what size is.."'''

    def add_generate_args(self, index=0, templ_index=0):
        self.node.answer_type = "TAG"
        self.node.tag_name = "SIZE"
        self.node._filters_args["mem_type"] = "REFERENCE_OBJECT"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        size_options = random.choice(
            ["what size is", "what size do you think is", "what is the size of"]
        )
        phrase = random.choice([size_options])

        return phrase


class AskColour(TemplateObject):
    '''This template object repesents questions of type: "what colour is.."'''

    def add_generate_args(self, index=0, templ_index=0):
        self.node.answer_type = "TAG"
        self.node.tag_name = "COLOUR"
        self.node._filters_args["mem_type"] = "REFERENCE_OBJECT"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(
            [
                "what colour is",
                "what colour do you think is",
                "what is the colour of",
                "what color is",
                "what color do you think is",
                "what is the color of",
            ]
        )

        return phrase


class AskIs(TemplateObject):
    '''This template object repesents questions of type: "is .."'''

    def add_generate_args(self, index=0, templ_index=0):
        self.node.answer_type = "EXISTS"
        self.node._filters_args["mem_type"] = "REFERENCE_OBJECT"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["is"])

        return phrase
