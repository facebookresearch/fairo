"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with the Location tree component.
"""
import random

from generate_utils import *
from tree_components import *
from .template_object import *

##########################
### LOCATION TEMPLATES ###
##########################


class MoveHere(TemplateObject):
    """This template object repesents the intent to move "here" and sets the location."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["location_type"] = SpeakerPos

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        node_template = get_template_names(self, templ_index)

        if node_template[-1] == "MoveHere":
            phrase = random.choice(
                [
                    "come back to me",
                    "move back to me",
                    "come to me",
                    "walk to me",
                    "come to where I am",
                    "move back to where I am",
                    "move to where I am",
                    "come back to where I am",
                ]
            )
        # For infinite loop
        elif node_template[-1] == "ConditionTypeNever":
            phrase = random.choice(
                [
                    "follow me",
                    "keep following me",
                    "follow me around",
                    "keep walking with me",
                    "can you follow me",
                    "please follow me",
                    "can you please follow me",
                ]
            )

        return phrase


class MoveHereCoref(TemplateObject):
    """This template object repesents location to be where the speaker is looking."""

    def add_generate_args(self, index=0, templ_index=0):
        self._word = random.choice(
            [
                "come here",
                "get over here",
                "come back here",
                "come back over here",
                "get over here",
                "move back here",
                "walk back here",
            ]
        )
        self.node._location_args["location_type"] = None
        self.node._location_args["coref_resolve"] = "yes"  # self._word.split()[-1]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._word


class Stand(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["relative_direction"] = True
        self.node._location_args["relative_direction_value"] = "UP"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["stand on", "go on top of", "go stand on top of"])
        template = get_template_names(self, templ_index)
        if template[-1] not in ["BlockObjectThis", "BlockObjectThat", "ThereTemplateCoref"]:
            phrase = " ".join([phrase, random.choice(["", "the"])]).strip()
        return phrase


class Down(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["relative_direction"] = True
        self.node._location_args["relative_direction_value"] = "DOWN"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["get down from", "come down from"])
        template = get_template_names(self, templ_index)
        if template[-1] not in ["BlockObjectThis", "BlockObjectThat", "ThereTemplateCoref"]:
            phrase = " ".join([phrase, random.choice(["", "the"])]).strip()
        return phrase


class LocationBlockObjectTemplate(TemplateObject):
    """This template object sets the location to be a reference to a blockobject"""

    def add_generate_args(self, index=0, templ_index=0):
        if (
            type(self.node._location_args["location_type"]) is list
            and self.node._location_args["location_type"][0] != BlockObject
        ):
            self.node._location_args["location_type"].append(BlockObject)
        else:
            self.node._location_args["location_type"] = [BlockObject]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        return description["block_object"]


class AroundString(TemplateObject):
    """This template object is used for the dance action and sets the direction
    of dance: CLOCKWISE / ANTICLOCKWISE"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["relative_direction"] = True
        self.node._location_args["relative_direction_value"] = random.choice(
            ["CLOCKWISE", "ANTICLOCKWISE"]
        )

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template = get_template_names(self, templ_index)
        if "HumanReplace" in template:
            return self.node._location_args["relative_direction_value"].lower()
        phrase = random.choice(["around"])
        return phrase


class Between(TemplateObject):
    '''This template object represents relative direction "Between"'''

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["relative_direction"] = True
        self.node._location_args["relative_direction_value"] = "BETWEEN"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        return description["relative_direction"]


class RelativeDirectionTemplate(TemplateObject):
    """This template object repesents that the location is relative to something."""

    def add_generate_args(self, index=0, templ_index=0):
        template = get_template_names(self, templ_index)

        self.node._location_args["relative_direction"] = True
        if index + 1 < len(template):
            self.node._location_args["additional_direction"] = []
            if template[index + 1] == "LocationBlockObjectTemplate":
                self.node._location_args["additional_direction"].extend(
                    ["INSIDE", "OUTSIDE", "BETWEEN"]
                )
            if template[index + 1] in ["LocationBlockObjectTemplate", "LocationMobTemplate"]:
                self.node._location_args["additional_direction"].append("NEAR")
                if "BETWEEN" not in self.node._location_args["additional_direction"]:
                    self.node._location_args["additional_direction"].append("BETWEEN")

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        return description["relative_direction"]


class ClimbDirectionTemplate(TemplateObject):
    """This template object repesents that the location is on top of something."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["relative_direction"] = True
        self.node._location_args["relative_direction_value"] = "UP"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        return description["relative_direction"]


class CoordinatesTemplate(TemplateObject):
    """This template object repesents that the location is absolute coordinates."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["location_type"] = Coordinates

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        coordinates_list = " ".join((description["coordinates"].split())[2:])
        return coordinates_list


class LocationMobTemplate(TemplateObject):
    """This template object repesents that location is a reference to a Mob."""

    def add_generate_args(self, index=0, templ_index=0):
        node_template = get_template_names(self, templ_index)
        # handle "that pig"
        if index >= 1 and node_template[index - 1] == "ThisTemplate":
            if type(self.node._location_args["location_type"]) is list:
                self.node._location_args["location_type"].append("SpeakerLookMob")
            else:
                self.node._location_args["location_type"] = ["SpeakerLookMob"]
        else:
            if type(self.node._location_args["location_type"]) is list and (
                self.node._location_args["location_type"][0] != Mob
            ):
                self.node._location_args["location_type"].append(Mob)
            else:
                self.node._location_args["location_type"] = [Mob]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        node_template = get_template_names(self, templ_index)
        description = self.node.args[arg_index].generate_description()
        mob_desc = description["mob"]
        # drop "the" from "follow that the pig"
        if ("mob_prefix" in mob_desc and index >= 1) and (
            (node_template[index - 1] == "ThisTemplate") or ("BlockObjectThat" in node_template)
        ):
            mob_desc = mob_desc.pop("mob_prefix")
        return description["mob"]


class HereTemplate(TemplateObject):
    """This template object repesents location as where the speaker is."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["location_type"] = SpeakerPos

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["where I am", "where I am standing"])
        return phrase


class HereTemplateCoref(TemplateObject):
    """This template object repesents location to be where the speaker is looking."""

    def add_generate_args(self, index=0, templ_index=0):
        self._word = random.choice(["here", "over here"])
        self.node._location_args["location_type"] = None
        self.node._location_args["coref_resolve"] = "yes"  # self._word.split()[-1]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._word


class ThereTemplate(TemplateObject):
    """This template object repesents location to be where the speaker is looking."""

    def add_generate_args(self, index=0, templ_index=0):
        self._word = random.choice(["where I am looking"])
        self.node._location_args["location_type"] = SpeakerLook

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._word


class ThereTemplateCoref(TemplateObject):
    """This template object repesents location to be where the speaker is looking."""

    def add_generate_args(self, index=0, templ_index=0):
        self._word = random.choice(["there", "over there"])
        self.node._location_args["location_type"] = None
        self.node._location_args["coref_resolve"] = "yes"  # self._word.split()[-1]

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._word


class YouTemplate(TemplateObject):
    """This template object repesents location to be where the agent is."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["location_type"] = AgentPos

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index)
        if "Dig" in template_names and "Under" in template_names:
            return "you"

        return random.choice(["where you are", "where you are standing"])


# NOTE: this is used for SpeakerLookMob type location, doesn't need coref resolution
class ThisTemplate(TemplateObject):
    """This template object repesents location to be where the agent is."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["location_type"] = SpeakerLook

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return random.choice(["this", "that"])


class StepsTemplate(TemplateObject):
    """This template object repesents that the location involves taking few "steps"."""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._location_args["steps"] = True
        template_name = get_template_names(self, templ_index)

        if (index + 1 < len(template_name)) and (
            template_name[index + 1] in ["ConditionCount", "NTimes"]
        ):
            self.node._location_args["location_type"] = False

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        description = self.node.args[arg_index].generate_description()
        return description["steps"]


# The following templates generate only text in the context of
# Location templates.


class At(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return "at"


class ALittle(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(["a little", "a bit"])
        return phrase


LOCATION_TEMPLATES = [
    LocationBlockObjectTemplate,
    RelativeDirectionTemplate,
    CoordinatesTemplate,
    LocationMobTemplate,
    HereTemplate,
    ThereTemplate,
    ThereTemplateCoref,
    YouTemplate,
    ThisTemplate,
    StepsTemplate,
    At,
    ALittle,
    Between,
]
