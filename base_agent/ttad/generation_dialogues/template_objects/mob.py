"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with the Mob tree component.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *

#####################
### MOB TEMPLATES ###
#####################


def set_mob_location(template_obj, location):
    template_obj.node._mob_args["mob_location"] = location


# Note: this is for "this pig", no coref resolution needed
class MobThis(TemplateObject):
    """This template object represents 'this' mob"""

    def add_generate_args(self, index=0, templ_index=0):
        set_mob_location(self, SpeakerLook)

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "this"


# Note: this is for "that pig", no coref resolution needed
class MobThat(TemplateObject):
    """This template object represents 'that' mob"""

    def add_generate_args(self, index=0, templ_index=0):
        set_mob_location(self, SpeakerLook)

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "that"


class MobLocation(TemplateObject):
    """This template object ensures that Mob has a location"""

    def add_generate_args(self, index=0, templ_index=0):
        set_mob_location(self, "ANY")

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        word = description["location"]
        return word


class MobName(TemplateObject):
    """This template object sets the argument type to be a Mob"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._arg_type = Mob

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_name = get_template_names(self, templ_index)
        description = self.node.args[arg_index].generate_description()

        if any(x in ["RepeatAll", "RepeatCount"] for x in template_name) and (
            "mob_prefix" in description
        ):
            return " ".join([description["mob_prefix"], description["mob"]])
        elif "Spawn" in template_name:
            mob_name = description["mob"]
            # append a/an randomly: "Spawn pig" / "Spawn a pig"
            return random.choice([mob_name, prepend_a_an(mob_name)])
        return description["mob"]


MOB_TEMPLATES = [MobThis, MobThat, MobLocation, MobName]
