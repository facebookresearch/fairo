"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains common template objects used across different templates.
"""
import random

from droidlet.dialog.ttad.generation_dialogues.generate_utils import *
from droidlet.dialog.ttad.generation_dialogues.tree_components import *
from .template_object import *

#######################
## COMMON TEMPLATES  ##
#######################


"""This template object represents the phrase: do X n times """


class NTimes(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        num_copies = random.choice(self.template_attr.get("count", range(1, 101)))
        self.num_copies = random.choice([str(num_copies), int_to_words(num_copies)])
        self.node._repeat_args["repeat_key"] = "FOR"
        self.node._repeat_args["repeat_count"] = self.num_copies

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        command = random.choice(["{} times"]).format(self.num_copies)
        return command
