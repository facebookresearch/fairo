"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with Undo command.
"""
import random

from .template_object import *

#####################
### UNDO TEMPLATES ##
#####################


class ActionBuild(TemplateObject):
    """This template object repesents that the target action is Build."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index=templ_index)
        if "Undo" in template_names:
            phrases = [
                "what you just built",
                "what you made",
                "the build action",
                "the construction",
                "what you built",
            ]
        elif "Stop" in template_names:
            phrases = ["building", "constructing", "completing"]
        elif "Dont" in template_names:
            phrases = [
                "build anymore",
                "build any more copies",
                "build anything",
                "make anything",
                "construct anything",
                "copy anything",
            ]
        elif "Resume" in template_names:
            phrases = ["building", "copying", "making copies"]
        self.words = random.choice(phrases)

        self.node.target_action_type = self.words

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.words


class ActionDestroy(TemplateObject):
    """This template object repesents that the target action is Destroy."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index=templ_index)
        if "Undo" in template_names:
            phrases = ["what you destroyed", "the destruction", "the destroy action"]
        elif "Stop" in template_names:
            phrases = ["destroying", "excavating", "destructing"]
        elif "Dont" in template_names:
            phrases = ["destroy anything", "destroy", "do the destroy action"]
        elif "Resume" in template_names:
            phrases = ["destroying", "excavating"]
        self.words = random.choice(phrases)

        self.node.target_action_type = self.words

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.words


class ActionFill(TemplateObject):
    """This template object repesents that the target action is Fill."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index=templ_index)
        if "Undo" in template_names:
            phrases = [
                "what you just filled",
                "filling",
                "the fill action",
                "the filling action",
                "filling that",
                "filling the hole",
            ]
        elif "Stop" in template_names:
            phrases = ["filling", "filling holes"]
        elif "Dont" in template_names:
            phrases = ["fill", "do the fill action"]
        elif "Resume" in template_names:
            phrases = ["filling"]
        self.words = random.choice(phrases)
        self.node.target_action_type = self.words

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.words


# NOTE(kavya): this should become a delete for undo tag. And How about for resume and stop ?
class ActionTag(TemplateObject):
    """This template object repesents that the target action is Tag."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index=templ_index)
        if "Stop" in template_names:
            command = random.choice(["tagging", "the tag action", "labeling"])
        elif "Dont" in template_names:
            command = random.choice(
                ["tag", "tag anything", "do any tagging", "do any labeling", "label anything"]
            )
        elif "Resume" in template_names:
            command = random.choice(["tagging", "labeling"])
        else:
            phrases = [
                "undo what you tagged",
                "undo the tagging",
                "undo the tag",
                "undo the tag action",
                "reset the tag action",
                "forget I tagged that",
                "forget that tag",
            ]
            phrase = random.choice(phrases)
            prefix = random.choice(["", random.choice(["can you", "please", "can you please"])])
            command = (" ".join([prefix, phrase])).strip()

        self.words = command
        self.node.target_action_type = self.words

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.words


class ActionDig(TemplateObject):
    """This template object repesents that the target action is Dig."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index=templ_index)
        if "Undo" in template_names:
            phrases = ["what you dug", "the digging", "the hole", "the dig action", "digging"]
        elif "Stop" in template_names:
            phrases = ["digging"]
        elif "Dont" in template_names:
            phrases = ["dig", "dig anything"]
        elif "Resume" in template_names:
            phrases = ["digging"]
        self.words = random.choice(phrases)
        self.node.target_action_type = self.words

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.words


class ActionMove(TemplateObject):
    """This template object repesents that the target action is Move."""

    def add_generate_args(self, index=0, templ_index=0):
        template_names = get_template_names(self, templ_index=templ_index)
        if "Stop" in template_names:
            phrases = ["walking", "moving"]
        elif "Dont" in template_names:
            phrases = ["walk", "move"]
        elif "Resume" in template_names:
            phrases = ["moving", "walking"]
        self.words = random.choice(phrases)
        self.node.target_action_type = self.words

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self.words
