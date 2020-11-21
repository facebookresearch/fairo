"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains generic template objects associated with Dilogue.
"""
from generate_utils import *
from tree_components import *
from .template_object import *


class Human(TemplateObject):
    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return None  # "human:"


class HumanReplace(TemplateObject):
    def add_generate_args(self, index=0, templ_index=0):
        self.node._replace = True
        if self.node._no_children:
            self.node._no_children = False

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return None  # "human:"
