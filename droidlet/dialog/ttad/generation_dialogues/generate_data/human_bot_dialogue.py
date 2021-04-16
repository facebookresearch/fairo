"""
Copyright (c) Facebook, Inc. and its affiliates.

Actions:
- GetMemory (filters, answer_type)
- PutMemory (filters, info_type)
Top-Level = {
    "dialogue_type":  {
        `action_type`: {Action}
        }
    }
e.g. {
    "get_memory" : {
        "filters" : {
            "type" : "action",
            "temporal" : "current"
        }
    }
    }

Action = {
    {arg_type}: {arg_dict}   # e.g. Move dict = {"Location": {Location}}
}
"""
from .action_node import *
from droidlet.dialog.ttad.generation_dialogues.tree_components import *


class GetMemory(ActionNode):
    """The BotCurrentAction
    __init__(): Pick a template from move_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'location_type','relative_direction' etc. for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("GetMemory", template, template_attr=template_attr)
        self._is_dialogue = True

    @classmethod
    def generate(cls, template=None, template_attr={}):
        get_mem_obj = GetMemory(template, template_attr)
        template = get_mem_obj.template

        get_mem_obj.ARG_TYPES = []
        get_mem_obj._no_children = False  # no ARG_TYPE if _no_children is True

        get_mem_obj._block_obj_args = Arguments(
            {
                "block_object_type": Object,
                "block_object_attributes": None,
                "block_object_location": False,
                "no_child": False,
                "repeat_key": None,
                "repeat_location": None,
                "coref_type": None,
            }
        )

        get_mem_obj._location_args = Arguments(
            {
                "location_type": "ANY",
                "relative_direction": False,
                "steps": None,
                "repeat_key": None,
            }
        )

        get_mem_obj._filters_args = Arguments(
            {
                "has_tag": None,
                "mem_type": None,
                "has_name": None,
                "block_object_attr": get_mem_obj._block_obj_args,
                "location_attr": get_mem_obj._location_args,
            }
        )

        get_mem_obj.answer_type = None

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        get_mem_obj.args = []
        if get_mem_obj._no_children:
            return get_mem_obj

        if (
            get_mem_obj._filters_args.values_updated
            or get_mem_obj._block_obj_args.values_updated
            or get_mem_obj._location_args.values_updated
        ):
            if get_mem_obj._block_obj_args.values_updated:
                get_mem_obj._filters_args["bo_updated"] = True
            if get_mem_obj._location_args.values_updated:
                get_mem_obj._filters_args["location_updated"] = True
            get_mem_obj.ARG_TYPES.append(Filters)

            get_mem_obj.args.append(Filters(**get_mem_obj._filters_args))

        return get_mem_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            for i, key in enumerate(templ):

                # get the text from template object
                item = key.generate_description(arg_index=0, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations


class PutMemory(ActionNode):
    """The BotCurrentAction
    __init__(): Pick a template from move_templates.py.
    generate(): Instantiate the template_objects in the template that populate the
                child arguments for classes in ARG_TYPES.
                For example: the template objects populate values of
                'location_type','relative_direction' etc. for Location
    _generate_description(): Generates the text description using the template ojects.
    """

    def __init__(self, template=None, template_attr={}):
        super().__init__("PutMemory", template, template_attr=template_attr)
        self._is_dialogue = True

    @classmethod
    def generate(cls, template=None, template_attr={}):
        put_mem_obj = PutMemory(template, template_attr)
        template = put_mem_obj.template

        put_mem_obj.ARG_TYPES = []
        put_mem_obj._no_children = False  # no ARG_TYPE if _no_children is True
        put_mem_obj._arg_type = BlockObject
        put_mem_obj._block_obj_args = Arguments(
            {
                "block_object_type": Object,
                "block_object_attributes": None,
                "block_object_location": False,
                "no_child": False,
                "repeat_key": None,
                "repeat_location": None,
                "coref_type": None,
            }
        )
        put_mem_obj._mob_args = Arguments(
            {"mob_location": None, "repeat_key": None, "repeat_location": None}
        )
        put_mem_obj._filters_args = Arguments(
            {
                "has_tag": None,
                "mem_type": None,
                "has_name": None,
                "block_object_attr": put_mem_obj._block_obj_args,
                "mob_attr": put_mem_obj._mob_args,
            }
        )

        put_mem_obj._upsert_args = Arguments(
            {
                "memory_type": None,
                "reward_value": None,
                "has_tag": None,
                "has_size": None,
                "has_colour": None,
            }
        )
        put_mem_obj.info_type = None

        # change default arguments for ARG_TYPE classes using the template_objects.
        for j, templ in enumerate(template):
            for i, t in enumerate(templ):
                if type(t) != str:
                    if callable(getattr(t, "add_generate_args", None)):
                        t.add_generate_args(index=i, templ_index=j)

        # Append the ARG_TYPE object with arguments, to generate the action tree
        put_mem_obj.args = []
        if put_mem_obj._no_children:
            return put_mem_obj

        if put_mem_obj._arg_type == Mob:
            put_mem_obj._filters_args["mob_updated"] = True
        elif put_mem_obj._block_obj_args.values_updated:
            put_mem_obj._filters_args["bo_updated"] = True

        put_mem_obj.ARG_TYPES.append(Filters)
        put_mem_obj.args.append(Filters(**put_mem_obj._filters_args))

        if put_mem_obj._upsert_args.values_updated:
            put_mem_obj.ARG_TYPES.append(Upsert)
            put_mem_obj.args.append(Upsert(**put_mem_obj._upsert_args))

        return put_mem_obj

    def _generate_description(self):
        """get the text form from template object"""
        generations = []

        for j, templ in enumerate(self.template):
            result = []
            for i, key in enumerate(templ):

                # get the text from template object
                item = key.generate_description(arg_index=0, index=i, templ_index=j)

                if not item:
                    continue
                # flatten nested dict
                if type(item) in [OrderedDict, dict]:
                    val_list = list(values_of_nested_dict(item))
                    result.extend(val_list)
                else:
                    result.append(item)
            generations.append(" ".join(result))

        return generations
