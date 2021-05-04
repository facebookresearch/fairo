"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains template objects associated with the Tag action
"""
import random

from ..generate_utils import *
from ..tree_components import *
from .template_object import *

#####################
### TAG TEMPLATES ###
#####################

tag_map = {"colour": COLOURS, "size": ABSTRACT_SIZE, "tag": TAG_ADJECTIVES}


class TagDesc(TemplateObject):
    """This template object has tags that can be used as adjectives.
    eg : this red shape is bright.
    """

    def add_generate_args(self, index=0, templ_index=0):
        self.node._upsert_args["memory_type"] = "TRIPLE"
        tag_name = random.choice(list(tag_map.keys()))
        tag = random.choice(tag_map[tag_name])
        # set the has_tag_name as the tag
        self.node._upsert_args["has_" + tag_name] = tag
        self._tag = tag

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return self._tag


class TagName(TemplateObject):
    """This template object repesents the name of a tag.
    eg: spider or big spider"""

    def add_generate_args(self, index=0, templ_index=0):
        self.node._upsert_args["memory_type"] = "TRIPLE"
        ALL_TAGS = ABSTRACT_SIZE + COLOURS + TAG_ADJECTIVES
        names_desc = [" ".join([desc, name]) for name in TAG_NAMES for desc in ALL_TAGS]
        names = random.choice([TAG_NAMES + names_desc])

        self._tag = random.choice(names)
        # add has_tag key
        self.node._upsert_args["has_tag"] = self._tag

    def generate_description(self, arg_index=0, index=0, templ_index=0):
        return prepend_a_an(self._tag)
