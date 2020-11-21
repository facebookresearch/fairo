"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import random

from generate_utils import *
from templates.templates import get_template


class ActionNode:
    """This class is an Action Node that represents the "Action" in the action_tree.

    A node can have a list of child nodes (ARG_TYPES) or a list of node types, it can be.
    (CHOICES).
    generate() : is responsible for initializing the ARG_TYPES and CHOICES.
    generate_description() : Generates the natural language description.
    to_dict() : Generates the action tree recursively using the children.
    """

    ARG_TYPES = None  # a list of child node types that need to be generated
    CHOICES = None  # a list of node types that can be substituted for this node

    def __init__(self, template_key, template=None, template_attr={}):
        self.args = None  # populated by self.generate()
        self.description = None  # populated by self.generate_description()
        if template_key != "Noop":
            self.template = get_template(template_key, self, template, template_attr)
        self._dialogue_type = "human_give_command"
        self._replace = None
        self._is_dialogue = False
        self._d = {}

    def generate_description(self):
        if self.description is None:
            self.description = self._generate_description()
        return self.description

    @classmethod
    def generate(cls, action_type=None, template_attr={}):
        if cls.ARG_TYPES:
            x = cls(template_attr=template_attr)
            x.args = []

            for arg in cls.ARG_TYPES:
                x.args.append(arg.generate())
            return x
        if cls.CHOICES:
            c = random.choice(action_type) if type(action_type) is list else action_type

            return c.generate(template_attr=template_attr)

        return cls(template_attr=template_attr)

    def __repr__(self):
        if self.args:
            return "<{} ({})>".format(type(self).__name__, ", ".join(map(str, self.args)))
        else:
            return "<{}>".format(type(self).__name__)

    def to_dict(self):
        """Generates the action dictionary for the sentence"""

        action_dict = {}

        action_description_split = [x.split() for x in self.description]
        if self.args:
            # update the tree recursively.
            for arg_type, arg in zip(self.ARG_TYPES, self.args):
                # Update the action_description for children to compute spans later
                arg._action_description = action_description_split

                arg_name = arg_type.__name__
                key = to_snake_case(arg_name)  # key name in dictionary is snake case

                # BlockObject and Mob are "reference_object" in the tree
                if arg_name in ["BlockObject", "Mob"]:
                    key = "reference_object"

                action_dict.update({key: arg.to_dict()})

        def substitute_with_spans(action_description_split, d):
            new_d = {}
            for k, v in d.items():
                if k.startswith("has"):
                    new_d[k] = find_span(action_description_split, v)
                else:
                    new_d[k] = v
            return new_d

        # Prune out unnecessary keys from the tree
        for attr, val in self.__dict__.items():
            if (
                not attr.startswith("_")
                and val not in (None, "", {})
                and attr not in ["args", "description", "template", "ARG_TYPES"]
            ):
                action_dict[attr] = val
                # for schematic key in Dig and reference_object in Fill
                if attr in ["schematic", "reference_object"]:
                    updated_val = substitute_with_spans(action_description_split, val)
                    action_dict[attr] = updated_val
                # Spans for keys : 'has_*' and repeat_count
                if (attr.startswith("has_")) or (
                    attr in ["repeat_count", "dance_type_name", "target_action_type"]
                ):
                    span = find_span(action_description_split, val)
                    action_dict[attr] = span
                if attr == "dance_type_name":
                    action_dict["dance_type"] = {attr: action_dict[attr]}
                    action_dict.pop(attr)

        action_name = type(self).__name__

        # For single word commands, add a blank block_object for Copy's tree
        if (action_name == "Copy") and ("reference_object" not in action_dict):
            action_dict["reference_object"] = {}

        # Copy is represented as a 'Build' action in the tree
        if action_name == "Copy":
            action_name = "Build"

        # Assign dialogue_type for classes that are dialogues
        if self._is_dialogue:
            self._dialogue_type = action_name

        # Assign replace key
        if self._replace:
            action_dict["replace"] = True

        self._d["dialogue_type"] = to_snake_case(self._dialogue_type, case="upper")

        # put action as a key for all actions
        if self._dialogue_type in ["human_give_command"]:
            action_dict["action_type"] = to_snake_case(action_name, case="upper")

            # move location inside reference_object for Fill action
            if action_name == "Fill":
                if "location" in action_dict:
                    if "reference_object" not in action_dict:
                        action_dict["reference_object"] = {}
                    action_dict["reference_object"]["location"] = action_dict["location"]
                    action_dict.pop("location")

            # fix reference object at action level
            if "reference_object" in action_dict:
                new_dict = {}
                val = action_dict["reference_object"]
                # if "repeat" in val:
                #     new_dict["repeat"] = val["repeat"]
                #     val.pop("repeat")
                if "special_reference" in val:
                    new_dict["special_reference"] = val["special_reference"]
                    val.pop("special_reference")
                new_dict["filters"] = val
                action_dict["reference_object"] = new_dict

            if "action_sequence" in self._d:
                self._d["action_sequence"].append(action_dict)
            else:
                self._d["action_sequence"] = [action_dict]
        else:
            # for get_memory
            if self._dialogue_type == "GetMemory":
                # fix layout of filters
                if "filters" in action_dict:
                    filters_dict = action_dict["filters"]
                    # fix the type
                    if "type" in filters_dict:
                        if filters_dict["type"] == "ACTION" and "action_type" in filters_dict:
                            filters_dict["memory_type"] = {
                                "action_type": filters_dict["action_type"]
                            }
                            filters_dict.pop("type")
                            filters_dict.pop("action_type")
                        else:
                            filters_dict["memory_type"] = filters_dict["type"]
                            filters_dict.pop("type")

                    # fix reference object in filters
                    if "reference_object" in filters_dict:
                        ref_obj_dict = filters_dict["reference_object"]
                        if "filters" in ref_obj_dict:
                            filters_dict.update(ref_obj_dict["filters"])
                            ref_obj_dict.pop("filters")
                        filters_dict.update(ref_obj_dict)
                        filters_dict.pop("reference_object")
                    # remove 'temporal' key
                    if filters_dict.get("temporal", None):
                        filters_dict.pop("temporal")
                    # replace old filters
                    action_dict["filters"] = filters_dict

                # fix answer_type and tag_name
                if action_dict.get("answer_type", None):
                    if action_dict["answer_type"] == "TAG" and "tag_name" in action_dict:
                        action_dict["filters"]["output"] = {"attribute": action_dict["tag_name"]}
                        action_dict.pop("tag_name")
                    else:
                        action_dict["filters"]["output"] = "memory"
                    action_dict.pop("answer_type")
            elif self._dialogue_type == "PutMemory":
                # fix layout of filters
                if "filters" in action_dict:
                    filters_dict = action_dict["filters"]
                    if "reference_object" in filters_dict:
                        ref_obj_dict = filters_dict["reference_object"]
                        if "filters" in ref_obj_dict:
                            filters_dict.update(ref_obj_dict["filters"])
                            ref_obj_dict.pop("filters")
                        filters_dict.update(ref_obj_dict)
                        filters_dict.pop("reference_object")
                action_dict["filters"] = filters_dict

            self._d.update(action_dict)

        return self._d
