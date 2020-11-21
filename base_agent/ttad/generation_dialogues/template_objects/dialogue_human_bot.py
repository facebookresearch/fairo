"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with human-bot dialogues.
"""
from generate_utils import *
from tree_components import *
from .template_object import *

action_reference_object_map = {
    "BUILD": "building",
    "DESTROY": "destroying",
    "SPAWN": "spawning",
    "MOVE": "following",
    "DIG": "digging",
    "FILL": "filling",
}


class QueryBotCurrentAction(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._filters_args["temporal"] = "CURRENT"
        self.node._filters_args["mem_type"] = "ACTION"
        self.node.answer_type = "TAG"
        self.node.tag_name = "action_name"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        question = random.choice(
            [
                "what are you doing",
                "tell me what are you doing",
                "what is your task",
                "tell me your task",
                "what are you up to",
            ]
        )

        return question


class QueryBot(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        curr_template = get_template_names(self, templ_index)
        if "MoveTarget" in curr_template:
            question = random.choice(["where are you", "tell me where you are"])
        elif "See" in curr_template:
            question = random.choice(
                [
                    "what is",
                    "what are the labels associated with",
                    "what are the categories of",
                    "tell me the properties of",
                ]
            )
        elif "CurrentLocation" in curr_template:
            question = random.choice(
                [
                    "where are you",
                    "tell me where you are",
                    "i do n't see you",
                    "i ca n't find you",
                    "are you still around",
                ]
            )
        else:
            question = random.choice(["what are you", "now what are you", "tell me what you are"])

        return question


class CurrentLocation(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._filters_args["temporal"] = "CURRENT"
        self.node._filters_args["mem_type"] = "AGENT"
        self.node.answer_type = "TAG"
        self.node.tag_name = "location"


class ActionReferenceObjectName(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._filters_args["temporal"] = "CURRENT"
        self.node._filters_args["mem_type"] = "ACTION"
        self.node._filters_args["action_type"] = random.choice(
            list(action_reference_object_map.keys())
        )
        self.node.answer_type = "TAG"
        self.node.tag_name = "action_reference_object_name"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        question = action_reference_object_map[self.node._filters_args["action_type"]]

        return question


class MoveTarget(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._filters_args["temporal"] = "CURRENT"
        self.node._filters_args["mem_type"] = "ACTION"
        self.node.answer_type = "TAG"
        self.node.tag_name = "move_target"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        question = random.choice(["heading", "off to", "going", "walking to", "heading over to"])
        self.node._action_name = question

        return question


class HumanReward(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._upsert_args["memory_type"] = "REWARD"


class PosReward(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._upsert_args["reward_value"] = "POSITIVE"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(
            [
                "good job",
                "cool",
                "that is really cool",
                "that is awesome",
                "awesome",
                "that is amazing",
                "that looks good",
                "you did well",
                "great",
                "good",
                "nice",
            ]
        )
        return phrase


class NegReward(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._upsert_args["reward_value"] = "NEGATIVE"

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        phrase = random.choice(
            [
                "that is wrong",
                "that was wrong",
                "that was completely wrong",
                "not that",
                "that looks horrible",
                "that is not what i asked",
                "that is not what i told you to do",
                "that is not what i asked for",
                "not what i told you to do",
                "you failed",
                "failure",
                "fail",
                "not what i asked for",
            ]
        )
        return phrase


class BotThank(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        reply = random.choice(["Thanks for letting me know."])
        return reply
