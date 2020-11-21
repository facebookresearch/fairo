"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file picks a template for a given action, at random.

The templates use template_objects as their children to help construct a sentence
and the dictionary.

TemplateObject is defined in template_objects.py
Each template captures how to phrase the intent. The intent is defined by the action
type.
"""
import copy
import random

from template_objects import *
from build_templates import *
from move_templates import *
from dig_templates import *
from destroy_templates import *
from copy_templates import *
from undo_templates import *
from fill_templates import *
from spawn_templates import *
from freebuild_templates import *
from dance_templates import *
from get_memory_templates import *
from put_memory_templates import *
from stop_templates import *
from resume_templates import *

template_map = {
    "Move": [MOVE_TEMPLATES, MOVE_WITH_CORRECTION],
    "Build": [BUILD_TEMPLATES, BUILD_WITH_CORRECTION, BUILD_INBUILT_COMPOSITE],
    "Destroy": [DESTROY_TEMPLATES, DESTROY_WITH_CORRECTION],
    "Dig": [DIG_TEMPLATES, DIG_WITH_CORRECTION],
    "Copy": [COPY_TEMPLATES, COPY_WITH_CORRECTION],
    "Undo": [UNDO_TEMPLATES],
    "Fill": [FILL_TEMPLATES, FILL_WITH_CORRECTION],
    "Spawn": [SPAWN_TEMPLATES],
    "Freebuild": [FREEBUILD_TEMPLATES, FREEBUILD_WITH_CORRECTION],
    "Dance": [DANCE_TEMPLATES, DANCE_WITH_CORRECTION],
    "GetMemory": [GET_MEMORY_TEMPLATES, ANSWER_WITH_CORRECTION],
    "PutMemory": [PUT_MEMORY_TEMPLATES, TAG_WITH_CORRECTION],
    "Stop": [STOP_TEMPLATES],
    "Resume": [RESUME_TEMPLATES],
}


def get_template(template_key, node, template=None, template_attr={}):
    """Pick a random template, given the action."""
    template_name = template_map[template_key]  # this will be a list right now
    if template_attr.get("dialogue_len", 0) == 1:
        template_name = template_name[0]
    elif template_attr.get("no_inbuilt_composites", False) == True and len(template_name) == 3:
        templates = []
        template_name = template_name[:2]
        for template_type in template_name:
            templates += template_type
        template_name = templates
    else:
        templates = []
        for template_type in template_name:
            templates += template_type
        template_name = templates

    if template is None:
        template = random.choice(template_name)
    template = copy.deepcopy(template)

    if not any(isinstance(i, list) for i in template):
        template = [template]

    for i, t in enumerate(template):
        for j, templ in enumerate(t):
            if type(templ) != str:
                template[i][j] = templ(node=node, template_attr=template_attr)

    return template
