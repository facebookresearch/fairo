"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects that generate only strings.
"""
import random

from ..generate_utils import *
from ..tree_components import *
from .template_object import *

################################
###  GENERIC TEXT TEMPLATES  ###
################################


"""The template objects under this section contribute only to the text /
description and do not alter the arguments of the parent node"""


class MadeOutOf(TemplateObject):
    def generate_description(self, arg_index, index, templ_index=0):
        if index == 0:
            return "out of"
        phrase = random.choice(["out of", "from", "using", "of", "with"])
        return phrase


class OnGround(TemplateObject):
    def generate_description(self, arg_index, index=0, templ_index=0):
        return "on the ground"


class All(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["all", "all the"])
        return phrase


class Every(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["every", "each"])
        return phrase


class The(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "the"


class To(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "to"


class LocationWord(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        word = random.choice(["loc", "location", "loc:", "location:"])
        return word


class Where(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "where"


class Is(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if index + 1 < len(template_names):
            if template_names[index + 1] == "TagName":
                return random.choice(["is", "looks like"])
            elif template_names[index + 1] == "TagDesc":
                return random.choice(["is", "looks"])
        return "is"


class SurroundWith(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return random.choice(["with", "using"])


class With(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(["with", "as"])
        return command


class Blocks(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["blocks", "block"])
        return phrase


class Squares(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["blocks", "squares", "block", "square", "tiles", "tile"])
        return phrase


class Wide(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = "wide"
        return phrase


class Long(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = "long"
        return phrase


class High(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = "high"
        return phrase


class Deep(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = "deep"
        return phrase


class InARow(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["in a row", "next to each other"])
        return phrase


class And(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = "and"
        return phrase


class Under(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "under"


class Times(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "times"


class Everything(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "everything"


class DownTo(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "down to"


class Find(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["find me", "find"])
        return phrase


class Up(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "up"


class ForMe(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["for me", ""])
        return phrase


class OfDimensionsPhrase(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        all_names = get_template_names(self, templ_index)
        phrases = ["of size", "of dimension", "of dimensions"]
        if "RepeatCount" in all_names:
            phrases.append("that are")
        else:
            phrases.append("that is")
        return random.choice(phrases)


class Please(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["please", ""])
        return phrase


class One(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        previous_template = get_template_names(self, templ_index - 1)
        phrase_list = ["one"]

        if any(x in ["RepeatCount", "RepeatAll"] for x in previous_template):
            phrase_list = ["ones"]

        phrase = random.choice(phrase_list)
        return phrase


class Thing(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["thing", "structure"])
        return phrase


class Using(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["using", "with"])
        return phrase


class Dont(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "do n't"
